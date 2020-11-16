from fastai.text.all import *
from fastai.text.data import _maybe_first
from transformers import *

class DropOutput(Callback):
    def after_pred(self):
        self.learn.pred = self.pred[0]
    
class TransformersTokenizer(Transform):
    def __init__(self, tokenizer, is_lm):
        self.tokenizer = tokenizer
        self.is_lm = is_lm

    def encodes(self, x):
        if self.is_lm:
            toks = self.tokenizer.tokenize(x)
            toks = self.tokenizer.convert_tokens_to_ids(toks)
            return TensorText(toks)
        else:
            return TensorText(
                self.tokenizer(x,
                               truncation=True,
                               padding='max_length',
                               max_length=512)['input_ids'])

    def decodes(self, x):
        return TitledStr(self.tokenizer.decode(x[1:-1].cpu().numpy()))


class MLMDataLoader(TfmdDL):
    "A `DataLoader` suitable for masked language modeling (MLM)"

    def __init__(self,
                 dataset,
                 lens=None,
                 cache=2,
                 bs=16,
                 num_workers=0,
                 **kwargs):
        #seq_len=510, 2 special token '[CLS]' and '[SEP]' will be added. Total 512 to meet bert sequence lenth.
        self.items = ReindexCollection(dataset, cache=cache, tfm=_maybe_first)
        #set_trace()
        self.seq_len = kwargs['seq_len'] - 2
        self.tok = kwargs['tok']
        self.cls = self.tok.cls_token_id
        self.sep = self.tok.sep_token_id
        self.mask = self.tok.mask_token_id
        #if lens is None: lens = fastai.text.data._get_lengths(dataset)
        if lens is None: lens = [len(o) for o in self.items]
        self.lens = ReindexCollection(lens,
                                      idxs=self.items.idxs)  #保持items和lens相同的顺序

        corpus = round_multiple(sum(lens), bs, round_down=True)
        #corput文本全长
        self.bl = corpus // bs  #bl stands for batch length
        self.n_batches = self.bl // (
            self.seq_len
        )  # + int(self.bl%seq_len!=0), abandon the last lenth (batch)
        #self.last_len = self.bl - (self.n_batches-1)*seq_len
        self.make_chunks()
        super().__init__(dataset=dataset,
                         bs=bs,
                         num_workers=num_workers,
                         **kwargs)
        self.n = self.n_batches * bs if len(
            self.items) > 1 else 1  # self.n for learner.predict

    def make_chunks(self):
        self.chunks = Chunks(self.items, self.lens)

    def shuffle_fn(self, idxs):
        self.items.shuffle()
        self.make_chunks()
        return idxs

    def random_word(self, txt):
        label = tensor([0] * (self.seq_len))
        masked_txt = txt
        for i in range(self.seq_len - 2):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                label[i] = txt[i]
                # 80% randomly change token to mask token
                if prob < 0.8:
                    masked_txt[i] = self.mask
                # 10% randomly change token to random token
                elif prob < 0.9:
                    masked_txt[i] = random.randrange(self.tok.vocab_size)
                # 10% randomly change token to current token, so no change.
        return masked_txt, label

    def create_item(self, seq):
        if seq >= self.n: raise IndexError
        st = (seq % self.bs) * self.bl + (seq // self.bs) * (self.seq_len)
        txt = self.chunks[st:st + self.seq_len] if self.n != 1 else self.items[
            0]  #self.dataset[0][0]
        if len(self.dataset) == 1:
            txt = torch.cat((tensor([self.cls]), txt, tensor([self.sep])))
            return (LMTensorText(txt), )
        masked_txt, label = self.random_word(txt)
        label = torch.cat((tensor([self.cls]), label, tensor([self.sep])))
        masked_txt = torch.cat(
            (tensor([self.cls]), masked_txt, tensor([self.sep])))
        return (LMTensorText(masked_txt), label)

    @delegates(TfmdDL.new)
    def new(self, dataset=None, seq_len=None, tok=None, **kwargs):
        lens = self.lens.coll if dataset is None else None
        seq_len = self.seq_len if seq_len is None else seq_len
        tok = self.tok if tok is None else tok
        return super().new(dataset=dataset,
                           lens=lens,
                           seq_len=seq_len,
                           tok=tok,
                           **kwargs)


class BertTextBlock(TransformBlock):
    "A `TransformBlock` for texts used in Bert"

    def __init__(self, tok_tfm, is_lm=False, **kwargs):
        type_tfms = [tok_tfm]
        seq_len = 512  #tok_tfm.tokenizer.model_max_length
        tok = tok_tfm.tokenizer
        return super().__init__(
            type_tfms=type_tfms,
            dl_type=MLMDataLoader if is_lm else SortedDL,
            dls_kwargs={
                'seq_len': seq_len,
                'tok': tok
            } if is_lm else {'before_batch': partial(pad_input, pad_idx=0)})


class BertTextDataLoaders(DataLoaders):
    "Basic wrapper around several `DataLoader`s with factory methods for NLP problems"

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_df(cls,
                df,
                model_name,
                path='.',
                valid_pct=0.2,
                seed=42,
                text_col=0,
                label_col=1,
                is_lm=False,
                **kwargs):
        "Create from `df` in `path` with `valid_pct`"
        tok = AutoTokenizer.from_pretrained(model_name)
        tok_tfm = TransformersTokenizer(tok, is_lm)
        blocks = [BertTextBlock(tok_tfm, is_lm)]
        if not is_lm: blocks.append(CategoryBlock)
        splitter = RandomSplitter(valid_pct, seed=seed)
        dblock = DataBlock(blocks=blocks,
                           get_x=ColReader("text"),
                           get_y=None if is_lm else ColReader(label_col),
                           splitter=splitter)
        #set_trace()
        return cls.from_dblock(dblock, df, path=path, **kwargs)


class BertTextLearner(Learner):
    "Basic class for a `Learner` in NLP."

    def save_encoder(self, file):
        "Save the encoder to `file` in the model directory"
        if rank_distrib(): return  # don't save if child proc
        encoder = get_model(self.model.base_model)
        torch.save(
            encoder.state_dict(),
            join_path_file(file, self.path / self.model_dir, ext='.pth'))

    def load_encoder(self, file, device=None):
        "Load the encoder `file` from the model directory, optionally ensuring it's on `device`"
        encoder = get_model(self.model.base_model)
        if device is None: device = self.dls.device
        distrib_barrier()
        wgts = torch.load(join_path_file(file,
                                         self.path / self.model_dir,
                                         ext='.pth'),
                          map_location=device)
        encoder.load_state_dict(wgts)
        self.freeze()
        return self


def bert_cls_splitter(m):
    "Split the classifier head from the backbone"
    groups = L([m.bert.embeddings] + [m.bert.encoder] + [m.bert.pooler] +
               [m.classifier])
    return groups.map(params)


def bert_lm_splitter(m):
    "Split the classifier head from the backbone"
    groups = L([m.bert.embeddings] + [m.bert.encoder] + [m.bert.pooler] +
               [m.cls])
    return groups.map(params)


def distilbert_cls_splitter(m):
    groups = L([m.distilbert.embeddings] + [m.distilbert.transformer] +
               [m.pre_classifier] + [m.classifier])
    return groups.map(params)


def distilbert_lm_splitter(m):
    groups = L([m.distilbert.embeddings] + [m.distilbert.transformer] +
               [m.vocab_transform] + [m.vocab_projector])
    return groups.map(params)


def albert_cls_splitter(m):
    groups = L([m.albert.embeddings] + [m.albert.encoder] + [m.albert.pooler] +
               [m.classifier])
    return groups.map(params)


def albert_lm_splitter(m):
    groups = L([m.albert.embeddings] + [m.albert.encoder] + [m.albert.pooler] +
               [m.predictions])
    return groups.map(params)


def roberta_cls_splitter(m):
    groups = L([m.roberta.embeddings] + [m.roberta.encoder] +
               [m.roberta.pooler] + [m.classifier])
    return groups.map(params)


def roberta_lm_splitter(m):
    groups = L([m.roberta.embeddings] + [m.roberta.encoder] +
               [m.roberta.pooler] + [m.lm_head])
    return groups.map(params)


splitters = {
    'bert_cls_splitter': bert_cls_splitter,
    'albert_cls_splitter': albert_cls_splitter,
    'distilbert_cls_splitter': distilbert_cls_splitter,
    'roberta_cls_splitter': roberta_cls_splitter,
    'bert_lm_splitter': bert_lm_splitter,
    'albert_lm_splitter': albert_lm_splitter,
    'distilbert_lm_splitter': distilbert_lm_splitter,
    'roberta_lm_splitter': roberta_lm_splitter
}


def AdamW(params, lr, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.01):
    params = [param for i in range(len(params)) for param in params[i]]
    return OptimWrapper(
        torch.optim.AdamW(params,
                          betas=(mom, sqr_mom),
                          eps=eps,
                          weight_decay=wd))
