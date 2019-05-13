# batch_adj_trajs = np.array(batch_adj_trajs)
# batch_node_trajs = np.array(batch_node_trajs)
# batch_ac_trajs = np.array(batch_ac_trajs)
# batch_smis_vec = np.array(batch_smis_vec)
# batch_adjs = np.reshape(batch_adj_trajs, [-1, ob_space['adj'].shape[0], ob_space['adj'].shape[1], ob_space['adj'].shape[2]])
# batch_nodes = np.reshape(batch_node_trajs, [-1, ob_space['node'].shape[0], ob_space['node'].shape[1], ob_space['node'].shape[2]])
# batch_acs = np.reshape(batch_ac_trajs, [-1, 4])
# batch_smis = np.reshape(batch_smis_vec, [-1, args.smi_max_length, len(env.smile_chars)])
# ob_experts, ac_experts, ori_smis = env.get_batch_expert_traj(optim_batchsize)  #env.get_expert(optim_batchsize, args.samples_num)
# samples = np.reshape(np.tile(np.expand_dims(samples, axis=1), [1, int(batch_smis.shape[0]/optim_batchsize), 1, 1]), [-1, 1, batch_node_trajs.shape[-1]])




# batch_data = np.random.choice(expert_seg, optim_batchsize)
# batch_adj_trajs, batch_node_trajs, batch_ac_trajs, batch_smis = make_batch(batch_data)
# batch_smis_vec = env.batch_smi2vec(args, batch_smis)
# samples = np.random.randn(optim_batchsize, 1, batch_node_trajs.shape[-1])
# loss_expert, loss_kl, g_expert = lossandgrad_seq_expert(batch_adj_trajs, batch_node_trajs, batch_smis_vec, samples, batch_ac_trajs)
