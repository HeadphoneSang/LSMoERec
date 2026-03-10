from recbole.quick_start.quick_start import run_recbole
from recbole.utils.encodeUtils import dict_to_table_str, EventType, emit_event
from recbole.utils.wechat import send_wecom_robot_msg

parameter_dict = {
    'learning_rate': 0.001,  #0.001
    'train_batch_size': 2048,
    'eval_batch_size': 2048,
    'train_neg_sample_args': None,  #
    'neg_sampling': None,
    'mask_ratio': 0.2,
    'hidden_size': 64,  #ML-1M 128
    'embedding_size': 64,
    'num_layers': 2,
    'n_heads': 8,
    'dropout_prob': 0.2,  #控制一开始的嵌入表示的丢弃  ML-1M:0.2
    'hidden_dropout_prob': 0.2,  #控制attention的结果的丢弃  ML-1M:0.2
    'attn_dropout_prob': 0.2,  #控制注意力系数的丢弃，暂时没用  ML-1M:0.2
    'hidden_act': 'gelu',
    'layer_norm_eps': 1e-12,
    'initializer_range': 0.02,
    'eval_args': {'split': {'LS': 'valid_and_test'}, 'order': 'TO', 'mode': 'fas100', 'group_by': 'user',
                  'neg_field': 'neg_sample_ids'},
    'topk': 10,
    'metrics': ['Recall', 'MRR', 'NDCG'],
    'valid_metric': 'NDCG@10',
    'moe_gate_t': 0.5,
    'align_lambda': 0.1,
    'spec_lambda': -0.1,
    'bal_lambda': 0.1,
    'time_b': 1.5,
    'kernel_mul': 2,
    'kernel_num': 2,  # The more kernel there are, the greater the impact of MMD on the differences among the samples.
    'conv_kernel_size': 3
}
train_valid_result = run_recbole(model='GLINTRU', dataset='games', config_dict=parameter_dict)
res_str = dict_to_table_str(train_valid_result)
webhook = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=6b65c26a-1314-4708-8fbf-03fa1ecb979e'
send_wecom_robot_msg(webhook, res_str)
