from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPreTrainedModel, BertModel, BertLMPredictionHead, BertForSequenceClassification
import torch
from torch import nn
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
import sys


class LOTClassModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        # MLM head is not trained
        for param in self.cls.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, pred_mode, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, mlm_labels=None, labels=None, template_tokens=None):
        if pred_mode == "inference":
            bert_outputs = self.bert(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds)
            last_hidden_states = bert_outputs[0]
            return last_hidden_states

        if pred_mode == "mlm_inference":
            bert_outputs = self.bert(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds)
            last_hidden_states = bert_outputs[0]
            logits = self.cls(last_hidden_states)
            return logits

        if pred_mode == "cl":
            cat_input_ids = torch.cat((input_ids, input_ids), dim=1).view(-1,input_ids.size(-1))
            cat_attention_mask = torch.cat((attention_mask, attention_mask), dim=1).view(-1,input_ids.size(-1))

            bert_outputs = self.bert(cat_input_ids,
                                     attention_mask=cat_attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds)
            last_hidden_states = bert_outputs[0]
            mask_idx = cat_input_ids == 103
            doc_embs = last_hidden_states[mask_idx]
            return simcse_loss(doc_embs)

        if pred_mode == "mlm":
            bert_outputs = self.bert(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds)
            last_hidden_states = bert_outputs[0]
            prediction_scores = self.cls(last_hidden_states)
            assert mlm_labels is not None
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            return masked_lm_loss

        # 自认为不太对
        # if pred_mode == "cl+mlm":
        #     cat_input_ids = torch.cat((input_ids, input_ids), dim=1).view(-1, input_ids.size(-1))
        #     cat_attention_mask = torch.cat((attention_mask, attention_mask), dim=1).view(-1, input_ids.size(-1))
        #     mlm_labels = torch.cat((mlm_labels, mlm_labels), dim=1).view(-1, input_ids.size(-1))
        #
        #     bert_outputs = self.bert(cat_input_ids,
        #                              attention_mask=cat_attention_mask,
        #                              token_type_ids=token_type_ids,
        #                              position_ids=position_ids,
        #                              head_mask=head_mask,
        #                              inputs_embeds=inputs_embeds)
        #     last_hidden_states = bert_outputs[0]
        #     mask_idx = cat_input_ids == 103
        #     doc_embs = last_hidden_states[mask_idx]
        #     cl_loss = simcse_loss(doc_embs)
        #
        #     prediction_scores = self.cls(last_hidden_states)
        #     assert mlm_labels is not None
        #     loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        #     mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        #
        #     return cl_loss, mlm_loss

        # 自认为是对的
        if pred_mode == "cl+mlm2":
            cat_input_ids = torch.cat((input_ids, input_ids), dim=1).view(-1, input_ids.size(-1))
            cat_attention_mask = torch.cat((attention_mask, attention_mask), dim=1).view(-1, input_ids.size(-1))
            mlm_labels = torch.cat((mlm_labels, mlm_labels), dim=1).view(-1, input_ids.size(-1))
            template_tokens = torch.cat((template_tokens, template_tokens), dim=1).view(-1, input_ids.size(-1))

            bert_outputs = self.bert(cat_input_ids,
                                     attention_mask=cat_attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds)
            last_hidden_states = bert_outputs[0]
            mask_idx = cat_input_ids == 103
            template_indicator = template_tokens == 1

            doc_emb_idx = mask_idx * template_indicator

            doc_embs = last_hidden_states[doc_emb_idx]
            cl_loss = simcse_loss(doc_embs)

            prediction_scores = self.cls(last_hidden_states)
            assert mlm_labels is not None
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

            return cl_loss, mlm_loss

        if pred_mode == "classifier":
            bert_outputs = self.bert(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds)
            last_hidden_states = bert_outputs[0]
            mask_idx = input_ids == 103
            mask_hidden_states = last_hidden_states[mask_idx]
            logits = self.classifier(mask_hidden_states)
            softmaxed_logits = torch.softmax(logits, dim=1)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                loss_fct = nn.KLDivLoss()
                one_hot_labels = torch.zeros(len(labels), self.num_labels).to(self.device).scatter_(1, labels.unsqueeze(-1), 1)
                loss = loss_fct(softmaxed_logits.log(), one_hot_labels.to(torch.float32))
                return logits, loss
            else:
                return logits

        if pred_mode == "raw_classifier":
            bert_outputs = self.bert(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds)
            last_hidden_states = bert_outputs[0]
            mask_hidden_states = last_hidden_states[:, 0, :]
            logits = self.classifier(mask_hidden_states)
            softmaxed_logits = torch.softmax(logits, dim=1)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                loss_fct = nn.KLDivLoss()
                one_hot_labels = torch.zeros(len(labels), self.num_labels).to(self.device).scatter_(1, labels.unsqueeze(-1), 1)
                loss = loss_fct(softmaxed_logits.log(), one_hot_labels.to(torch.float32))
                return logits, loss
            else:
                return logits



        # if pred_mode == "classification":
        #     trans_states = self.dense(last_hidden_states)
        #     trans_states = self.activation(trans_states)
        #     trans_states = self.dropout(trans_states)
        #     logits = self.classifier(trans_states)
        # elif pred_mode == "mlm":
        #     logits = self.cls(last_hidden_states)
        # else:
        #     sys.exit("Wrong pred_mode!")

def simcse_loss(y_pred):
    idxs = torch.arange(0, y_pred.size()[0]).to(0)
    idxs_ = idxs + 1 - idxs % 2 * 2
    y_pred = torch.nn.functional.normalize(y_pred, p=2, dim=1)
    similarities = torch.mm(y_pred, torch.transpose(y_pred, 0, 1))
    similarities = similarities - torch.eye(y_pred.size()[0]).to(0) * 1e10
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(similarities, idxs_)


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):

        return cl_forward(self, self.bert,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mlm_input_ids=mlm_input_ids,
            mlm_labels=mlm_labels,
        )


class BertForClassification(BertForSequenceClassification):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )