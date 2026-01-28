from recbole.quick_start.quick_start import run_recbole

parameter_dict = {
    'learning_rate': 0.001,  #0.001
    'train_batch_size': 2048,
    'eval_batch_size': 2048,
    'train_neg_sample_args': None,  #
    'neg_sampling': None,
    'mask_ratio': 0.2,
    'hidden_size': 64,
    'embedding_size': 64,
    'num_layers': 2,
    'n_heads': 8,
    'dropout_prob': 0.5,  #控制一开始的嵌入表示的丢弃
    'hidden_dropout_prob': 0.5,  #控制attention的结果的丢弃
    'attn_dropout_prob': 0.5,  #控制注意力系数的丢弃，暂时没用
    'hidden_act': 'gelu',
    'layer_norm_eps': 1e-12,
    'initializer_range': 0.02,
    'eval_args': {'split': {'LS': 'valid_and_test'}, 'order': 'TO', 'mode': 'fas100', 'group_by': 'user',
                  'neg_field': 'neg_sample_ids'},
    'topk': 10,
    'metrics': ['Recall', 'MRR', 'NDCG'],
    'valid_metric': 'NDCG@10'
}
run_recbole(model='LSMoERec', dataset='beauty', config_dict=parameter_dict)
