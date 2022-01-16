import numpy as np


def mask(input_tokens, input_mask_index, input_mask_type, mask_id):
    output_tokens = []
    mask_label = []
    mask_pos = []
    mask_pos_2 = []
    r_flag = []

    seq_len = -1
    actual_seq_len = []
    for sent_index, sent in enumerate(input_tokens):
        seq_len = max(seq_len, len(sent))
    
    if input_mask_index != -1:
        # under this situation, seq_len must be 3
        assert seq_len == 3
        for sent_index, sent in enumerate(input_tokens):
            assert len(sent) == 3
            assert input_mask_index == 0 or input_mask_index == 2 
            mask_label.append(sent[input_mask_index])
            mask_pos.append(sent_index * seq_len + input_mask_index)
            mask_pos_2.append(sent_index * seq_len + 2 - input_mask_index)
            if input_mask_index == 0:
                r_flag.append(-1)
            elif input_mask_index == 2:
                r_flag.append(1)
            else:
                assert 0
            sent_out = sent[:]
            sent_out[input_mask_index] = mask_id
            actual_seq_len.append(len(sent_out))
            output_tokens.append(sent_out)
    else:
        for sent_index, sent in enumerate(input_tokens):
            mask_type = input_mask_type[sent_index]
            if not mask_type.startswith("MASK"):
                assert 0
            token_index = int(mask_type.split("_")[-1])
            mask_label.append(sent[token_index])
            mask_pos.append(sent_index * seq_len + token_index)
            if seq_len == 3:
                mask_pos_2.append(sent_index * seq_len + 2 - token_index)
                if token_index == 0:
                    r_flag.append(-1)
                elif token_index == 2:
                    r_flag.append(1)
                else:
                    assert 0
            sent_out = sent[:]
            sent_out[token_index] = mask_id
            actual_seq_len.append(len(sent_out))
            if len(sent_out) < seq_len:
                sent_out += [0] * (seq_len - len(sent_out))
            output_tokens.append(sent_out)

    mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
    mask_pos_2 = np.array(mask_pos_2).astype("int64").reshape([-1, 1])
    r_flag = np.array(r_flag).astype("int64").reshape([-1, 1])
    output_tokens = np.array(output_tokens).astype("int64").reshape([-1, seq_len, 1])

    if mask_pos_2.shape[0] == 0:
        mask_pos_2 = None 
    if r_flag.shape[0] == 0:
        r_flag = None

    if seq_len == 3:
        output_mask = np.array([[1] * seq_len for _ in range(output_tokens.shape[0])])
        output_mask = np.expand_dims(output_mask, axis=-1).repeat(seq_len ,axis=-1).astype("int64")
    else:
        output_mask = np.array([[0] * seq_len for _ in range(seq_len)])
        for i in range(0, seq_len):
            output_mask[i][i] = 1
            if i % 2 == 0:
                if i == 0:
                    output_mask[i][i+1] = 1
                    output_mask[i][i+2] = 1
                elif i == seq_len - 1:
                    output_mask[i][i-1] = 1
                    output_mask[i][i-2] = 1
                else:
                    output_mask[i][i+1] = 1
                    output_mask[i][i+2] = 1
                    output_mask[i][i-1] = 1
                    output_mask[i][i-2] = 1
            else:
                output_mask[i][i-1] = 1
                output_mask[i][i+1] = 1
        output_mask = output_mask.reshape([-1, seq_len, seq_len]).repeat(output_tokens.shape[0], axis=0)
        for i in range(output_tokens.shape[0]):
            if actual_seq_len[i] < seq_len:
                output_mask[i][actual_seq_len[i]:][:] = 0
                output_mask[i][actual_seq_len[i]-1][actual_seq_len[i]:] = 0
        output_mask = output_mask.astype("int64")
    return output_tokens, mask_label, mask_pos, output_mask, mask_pos_2, r_flag


def prepare_batch_data(examples, mask_index, mask_id):
    batch_src_ids = examples
    batch_mask_type = None
    if mask_index == -1:
        batch_src_ids = [example[:-1] for example in examples]
        batch_mask_type = [example[-1] for example in examples]

    out, mask_label, mask_pos, out_mask, mask_pos_2, r_flag = mask(input_tokens=batch_src_ids, input_mask_index=mask_index,
                                               input_mask_type=batch_mask_type, mask_id=mask_id)
    return [out, out_mask, mask_label, mask_pos, mask_pos_2, r_flag]
