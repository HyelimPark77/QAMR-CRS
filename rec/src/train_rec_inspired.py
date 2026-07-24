import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_dbpedia_inspired import DBpedia, Co_occurrence, text_sim, image_sim
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import MMPrompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def serialize_selected_evidence(selected_evidence):
    if selected_evidence is None:
        return None

    serialized = {}
    for name, payload in selected_evidence.items():
        serialized[name] = {
            'entity_ids': make_json_safe(payload['entity_ids']),
            'scores': make_json_safe(payload['scores']),
        }
    return serialized


def write_evidence_log(log_file, split, epoch, step, router_weights, selected_evidence, labels):
    if log_file is None:
        return

    record = {
        'split': split,
        'epoch': epoch,
        'step': step,
        'router_weights': router_weights.mean(dim=0).detach().cpu().tolist() if router_weights is not None else None,
        'selected_evidence': serialize_selected_evidence(selected_evidence),
        'labels': make_json_safe(labels),
    }
    log_file.write(json.dumps(record, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=222, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='inspired_model', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    parser.add_argument("--dataset", type=str, default='inspired', help="A file containing all data.")
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--use_resp", action="store_true")
    parser.add_argument("--context_max_length", type=int,default=200, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int,default=200)
    parser.add_argument("--entity_max_length", type=int,default=32, help="max entity length in dataset.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--tokenizer", type=str,default='hf_models/DialoGPT-small')
    parser.add_argument("--text_tokenizer", type=str,default='hf_models/roberta-base')
    parser.add_argument("--model", type=str, default='hf_models/DialoGPT-small',help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str,default='hf_models/roberta-base')
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    parser.add_argument("--n_prefix_rec", type=int,default=10)
    parser.add_argument("--routing_beta", type=float, default=0.0,
                        help="Weight for static fused entity evidence in residual routing.")
    parser.add_argument("--learnable_routing_beta", action="store_true",
                        help="Learn the residual routing beta initialized from --routing_beta.")
    parser.add_argument("--entity_aware_routing", action="store_true",
                        help="Route modality evidence separately for each mentioned entity.")
    parser.add_argument("--entropy_lambda", type=float, default=0.0,
                        help="Weight for router regularization.")
    parser.add_argument("--router_balance_loss", choices=["instance", "batch"], default="instance",
                        help="Use instance-level negative entropy or batch-level load balancing.")
    parser.add_argument("--contrastive_lambda", type=float, default=1e-4,
                        help="Weight for the token/entity contrastive loss.")
    parser.add_argument("--query_aware_scoring", action="store_true",
                        help="Use router-conditioned entity embeddings for final recommendation scores.")
    parser.add_argument("--prompt_encoder", type=str,default='rec/src/pre-trained-inspired/final')
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=64,help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int,default=33)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    for attr in ("output_dir", "tokenizer", "text_tokenizer", "model", "text_encoder", "prompt_encoder"):
        value = getattr(args, attr)
        if value is not None and not os.path.isabs(value):
            setattr(args, attr, str(PROJECT_ROOT / value))
    config = vars(args)
    accelerator = Accelerator(device_placement=False, mixed_precision="fp16" if args.fp16 else "no")
    device = accelerator.device
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.makedirs(PROJECT_ROOT / 'log', exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(str(PROJECT_ROOT / 'log' / f'{local_time}.log'), level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)
        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None
    if accelerator.is_local_main_process:
        evidence_log = open(PROJECT_ROOT / 'log' / f'evidence_rec_{local_time}.jsonl', 'w', buffering=1)
    else:
        evidence_log = None
    if args.seed is not None:
        set_seed(args.seed)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)
    train_dataset = CRSRecDataset(
        dataset=args.dataset, split='train', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    co = Co_occurrence(dataset=args.dataset, split='train', debug=args.debug ,all_items = kg['item_ids'],entity_max_length=args.entity_max_length,n_entity=kg['num_entities'] ).get_entity_co_info()
    text_simi  = text_sim(pad_entity_id=kg['pad_entity_id']).get_entity_ts_info()
    image_simi = image_sim(pad_entity_id=kg['pad_entity_id']).get_entity_is_info()
    shot_len = int(len(train_dataset) * args.shot)
    train_dataset = random_split(train_dataset, [shot_len, len(train_dataset) - shot_len])[0]
    assert len(train_dataset) == shot_len
    valid_dataset = CRSRecDataset(
        dataset=args.dataset, split='valid', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    test_dataset = CRSRecDataset(
        dataset=args.dataset, split='test', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    data_collator = CRSRecDataCollator(
        tokenizer=tokenizer, device=device, debug=args.debug,
        context_max_length=args.context_max_length, entity_max_length=args.entity_max_length,
        pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    prompt_encoder = MMPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],edge_index_c = co['edge_index_c'],edge_index_t_s = text_simi['edge_index_t_s'],edge_index_i_s = image_simi['edge_index_i_s'],idx_to_id = text_simi['idx_to_id'],
        n_prefix_rec=args.n_prefix_rec,
        routing_beta=args.routing_beta,
        learnable_routing_beta=args.learnable_routing_beta,
        entity_aware_routing=args.entity_aware_routing,
        pad_entity_id=kg['pad_entity_id'],
    )
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)
    fix_modules = [model, text_encoder]
    for module in fix_modules:
        module.requires_grad_(False)
    modules = [prompt_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    evaluator = RecEvaluator()
    prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test examples = {len(test_dataset)}")
    logger.info(f"  Num valid examples = {len(valid_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)
    best_epoch = None

    for epoch in range(args.num_train_epochs):
        train_loss = []
        prompt_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds,loss_cl = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True,
                use_rec_prefix=True
            )
            router_weights = prompt_encoder.get_last_router_weights()
            selected_evidence = prompt_encoder.get_last_selected_evidence()
            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds_for_scoring(
                router_weights if args.query_aware_scoring else None
            )
            write_evidence_log(
                evidence_log,
                split='train',
                epoch=epoch,
                step=step,
                router_weights=router_weights,
                selected_evidence=selected_evidence,
                labels=batch['context']['rec_labels'],
            )
            loss = model(**batch['context'], rec=True).rec_loss / args.gradient_accumulation_steps
            if args.contrastive_lambda:
                loss = loss + loss_cl * args.contrastive_lambda
            router_reg_loss = prompt_encoder.get_last_router_entropy_loss()
            if args.router_balance_loss == "batch":
                router_reg_loss = prompt_encoder.get_last_router_balance_loss()
            if args.entropy_lambda and router_reg_loss is not None:
                loss = loss + router_reg_loss * args.entropy_lambda
            accelerator.backward(loss)
            train_loss.append(float(loss))
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                if run:
                    log_payload = {'loss': np.mean(train_loss) * args.gradient_accumulation_steps}
                    if router_weights is not None:
                        mean_router = router_weights.mean(dim=0)
                        log_payload.update({
                            'router/kg': float(mean_router[0]),
                            'router/co': float(mean_router[1]),
                            'router/text': float(mean_router[2]),
                            'router/image': float(mean_router[3]),
                        })
                    if router_reg_loss is not None:
                        log_payload['router/reg_loss'] = float(router_reg_loss.detach().cpu())
                    if getattr(prompt_encoder, "learnable_routing_beta", False):
                        beta = prompt_encoder.get_routing_beta()
                        log_payload['router/beta'] = float(beta.detach().cpu())
                    run.log(log_payload)
            if completed_steps >= args.max_train_steps:
                break

        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')
        del train_loss, batch
        valid_loss = []
        prompt_encoder.eval()
        for batch in tqdm(valid_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds,loss_cl = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_rec_prefix=True
                )
                write_evidence_log(
                    evidence_log,
                    split='valid',
                    epoch=epoch,
                    step=0,
                    router_weights=prompt_encoder.get_last_router_weights(),
                    selected_evidence=prompt_encoder.get_last_selected_evidence(),
                    labels=batch['context']['rec_labels'],
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds_for_scoring(
                    prompt_encoder.get_last_router_weights() if args.query_aware_scoring else None
                )
                outputs = model(**batch['context'], rec=True)
                valid_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits[:, kg['item_ids']]
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        valid_report = {}
        for k, v in report.items():
            if k != 'count':
                valid_report[f'valid/{k}'] = v / report['count']
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            accelerator.unwrap_model(prompt_encoder).save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            best_epoch = epoch
            logger.info(f'new best model with valid/{metric}={best_metric} at epoch {best_epoch}')

    if best_epoch is None:
        raise RuntimeError('No validation checkpoint was selected.')

    accelerator.wait_for_everyone()
    unwrapped_prompt_encoder = accelerator.unwrap_model(prompt_encoder)
    unwrapped_prompt_encoder.load(best_metric_dir)
    logger.info(
        f'loaded validation-selected checkpoint from epoch {best_epoch} '
        f'with valid/{metric}={best_metric}'
    )

    # Evaluate the test split once, using the validation-selected checkpoint.
    test_loss = []
    prompt_encoder.eval()
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds,loss_cl = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True,
                use_rec_prefix=True
            )
            write_evidence_log(
                evidence_log,
                split='test',
                epoch=best_epoch,
                step=0,
                router_weights=prompt_encoder.get_last_router_weights(),
                selected_evidence=prompt_encoder.get_last_selected_evidence(),
                labels=batch['context']['rec_labels'],
            )
            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds_for_scoring(
                prompt_encoder.get_last_router_weights() if args.query_aware_scoring else None
            )

            outputs = model(**batch['context'], rec=True)
            test_loss.append(float(outputs.rec_loss))
            logits = outputs.rec_logits[:, kg['item_ids']]
            ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
            ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
            labels = batch['context']['rec_labels']
            evaluator.evaluate(ranks, labels)

    report = accelerator.gather(evaluator.report())
    for k, v in report.items():
        report[k] = v.sum().item()

    test_report = {}
    for k, v in report.items():
        if k != 'count':
            test_report[f'test/{k}'] = v / report['count']
    test_report['test/loss'] = np.mean(test_loss)
    test_report['epoch'] = best_epoch
    logger.info(f'{test_report}')
    if run:
        run.log(test_report)
    evaluator.reset_metric()

    final_dir = os.path.join(args.output_dir, 'final')
    unwrapped_prompt_encoder.save(final_dir)
    logger.info(f'saved validation-selected model to {final_dir}')
    if evidence_log is not None:
        evidence_log.close()
