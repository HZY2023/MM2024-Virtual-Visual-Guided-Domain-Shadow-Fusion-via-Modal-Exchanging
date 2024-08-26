# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from tensorboardX import SummaryWriter

writer = SummaryWriter("./generate-fig")
import math
from typing import Any, Dict, List, NamedTuple, Optional
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from torch import Tensor
import os
import gc
import re
from tqdm import tqdm

from fairseq.modules.transformer_layer import TransformerEncoderLayer_hallucination

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("transformer")
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe(
                'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--img_feature_dim', type=int, default=2048,
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--pre_mix', type=bool, default=True,
                            help='if True, dont scale embeddings')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            src_tokens,
            src_lengths,
            # multimodel_graph,
            prev_output_tokens,
            src_img_features=None,
            cls_input: Optional[Tensor] = None,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_img_features_location: Optional[List] = None):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        # print('src_graphs:',src_graphs.size())
        if src_img_features is not None:
            encoder_out = self.encoder(
                src_tokens,
                src_lengths=src_lengths,
                src_img_features=src_img_features,
                # multimodel_graph=multimodel_graph,
                cls_input=cls_input,
                return_all_hiddens=return_all_hiddens,
                src_img_features_location=src_img_features_location
            )
        else:
            encoder_out = self.encoder(
                src_tokens,
                src_lengths=src_lengths,
                cls_input=cls_input,
                return_all_hiddens=return_all_hiddens
            )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        if src_img_features is not None:
            return decoder_out, encoder_out
        else:
            return decoder_out, encoder_out


@register_model("transformer_align")
class TransformerAlignModel(TransformerModel):
    """
    See "Jointly Learning to Align and Translate with Transformer
    Models" (Garg et al., EMNLP 2019).
    """

    def __init__(self, encoder, decoder, args):
        super().__init__(args, encoder, decoder)
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer
        self.full_context_alignment = args.full_context_alignment

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(TransformerAlignModel, TransformerAlignModel).add_args(parser)
        parser.add_argument('--alignment-heads', type=int, metavar='D',
                            help='Number of cross attention heads per layer to supervised with alignments')
        parser.add_argument('--alignment-layer', type=int, metavar='D',
                            help='Layer number which has to be supervised. 0 corresponding to the bottommost layer.')
        parser.add_argument('--full-context-alignment', type=bool, metavar='D',
                            help='Whether or not alignment is supervised conditioned on the full target context.')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        transformer_align(args)

        transformer_model = TransformerModel.build_model(args, task)
        return TransformerAlignModel(
            transformer_model.encoder, transformer_model.decoder, args
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        return self.forward_decoder(prev_output_tokens, encoder_out)

    def forward_decoder(
            self,
            prev_output_tokens,
            encoder_out=None,
            incremental_state=None,
            features_only=False,
            **extra_args,
    ):
        attn_args = {
            "alignment_layer": self.alignment_layer,
            "alignment_heads": self.alignment_heads,
        }
        decoder_out = self.decoder(prev_output_tokens, encoder_out, **attn_args)

        if self.full_context_alignment:
            attn_args["full_context_alignment"] = self.full_context_alignment
            _, alignment_out = self.decoder(
                prev_output_tokens,
                encoder_out,
                features_only=True,
                **attn_args,
                **extra_args,
            )
            decoder_out[1]["attn"] = alignment_out["attn"]

        return decoder_out


EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Tensor),  # B x T
        ("encoder_embedding", Tensor),
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("hallucination_train", Optional[Tensor]),
        ("hallucination_test", Optional[Tensor]),
        ("src_img_features", Optional[Tensor]),
    ],
)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.args = args
        self.dropout = args.dropout
        self.Gating = GatingMechanism(args)
        self.encoder_layerdrop = args.encoder_layerdrop
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.img_feature_dim = args.img_feature_dim
        self.img_fc = Linear(self.img_feature_dim, embed_dim)
        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )

        self.layers_hallucination = nn.ModuleList([])
        self.layers_hallucination.extend(
            [TransformerEncoderLayer_hallucination(args) for i in range(args.encoder_layers)]
        )

        self.mlp_model = MLP(input_dim=2048, output_dim=embed_dim, dropout_prob=0.1)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.selective_gating_text = SelectiveGating_text(embed_dim)
        self.selective_gating_vision = SelectiveGating_vision(embed_dim)
        self.BalancedFeatureSelector = BalancedFeatureSelector(embed_dim, embed_dim)
        self.vision_text_space = TextVisualFeatureMapper(embed_dim, embed_dim, 128)

        self.TextSelfAttention = TextSelfAttention(embed_dim)
        self.VisionSelfAttention = VisualSelfAttention(embed_dim)

        self.VisualBatchNorm = VisualBatchNorm(embed_dim)

        self.cross_gating = GatingMechanism(args)

        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.weight3 = nn.Parameter(torch.randn(1))


    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def getBinaryTensor(self, i, boundary):
        one_matrix = torch.ones_like(i)
        zero_matrix = torch.zeros_like(i)

        return torch.where(i > boundary, one_matrix, zero_matrix)

    def kl_divergence_loss(self, x_change_before, x_change_after):

        if x_change_before.size(0) != x_change_after.size(0):
            x_change_after = x_change_before.transpose(0, 1)
            x_change_before = x_change_before.transpose(0, 1)
            assert x_change_before.size(0) == x_change_after.size(0)

        probs_x_change_before = F.softmax(x_change_before, dim=-1)
        probs_x_change_after = F.softmax(x_change_after, dim=-1)

        log_probs_x_change_after = torch.log(probs_x_change_after + 1e-6)

        kl_div = F.kl_div(log_probs_x_change_after, probs_x_change_before, reduction='batchmean')

        kl_div = torch.nan_to_num(kl_div)

        return kl_div

    def probability_sampling(self, text_probs, visual_probs, num_samples, src_tokens, pad_token_id):
        text_probs = text_probs.clone()
        text_probs = 1.0 - text_probs
        text_probs = text_probs / text_probs.sum(dim=1, keepdim=True)
        visual_probs[visual_probs < 0.01] = 0
        visual_probs = visual_probs / visual_probs.sum(dim=1, keepdim=True)

        text_probs = text_probs.squeeze(-1)
        sampled_text_indices = torch.multinomial(text_probs, num_samples=num_samples, replacement=True)

        sampled_visual_indices = torch.multinomial(visual_probs.squeeze(-1), num_samples=num_samples, replacement=True)

        return sampled_text_indices, sampled_visual_indices

    def normalize_and_standardize(self, features):
        min_val = features.min()
        max_val = features.max()
        normalized_features = (features - min_val) / (max_val - min_val + 1e-3)

        mean = normalized_features.mean()
        std = normalized_features.std()

        if std < 1e-3:
            return features

        standardized_features = (normalized_features - mean) / (std + 1e-4)

        return standardized_features

    def exchange_tokens(self, text_features, visual_features, text_indices, visual_indices):
        batch_size, length, dim = text_features.size()

        text_features_flat = text_features.reshape(batch_size * length, dim)
        visual_features_flat = visual_features.reshape(batch_size * 49, dim)

        text_indices_flat = text_indices + (torch.arange(batch_size).unsqueeze(1) * length).cuda()
        visual_indices_flat = visual_indices + (torch.arange(batch_size).unsqueeze(1) * 49).cuda()

        text_indices_flat = text_indices_flat.view(-1)
        visual_indices_flat = visual_indices_flat.view(-1)

        temp = text_features_flat[text_indices_flat].clone()
        text_features_flat[text_indices_flat] = visual_features_flat[visual_indices_flat]
        visual_features_flat[visual_indices_flat] = temp

        return text_features_flat.reshape(batch_size, length, dim), visual_features

    def forward(
            self,
            src_tokens,
            src_lengths,
            src_img_features=None,
            # multimodel_graph,
            cls_input: Optional[Tensor] = None,
            return_all_hiddens: bool = False,
            sample=None,
            idx=None,
            src_img_features_location=None
    ):

        torch.autograd.set_detect_anomaly(True)
        all_src_img_features = []

        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)

        if src_img_features is not None:
            ###########image_feature_fc##############
            src_img_features = self.mlp_model(src_img_features)
            #########################################

        # B x T x C -> T x B x C
        # x = (weighted_context*x).transpose(0, 1)
        x = x.transpose(0, 1)
        x_hallucination = x
        batch_len = src_lengths[0].item()

        encoder_padding_mask_text = src_tokens.eq(self.padding_idx)
        encoder_padding_mask_hallucination = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        ##########batch normalization#########
        if src_img_features is not None:
            src_img_features = self.VisualBatchNorm(src_img_features)
        ####################################

        encoder_states_hallucination = []

        for idx, layer_hallucination in enumerate(self.layers_hallucination):
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                hallucination = layer_hallucination(x_hallucination, encoder_padding_mask_text,
                                                    encoder_padding_mask_hallucination)

                encoder_states_hallucination.append(hallucination)

        x_oral_representation = []
        x_change_representation = []

        for idx, layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if x.size(0) == batch_len:
                x = x.transpose(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                if src_img_features is not None:
                    ############## progressive modality-mixing module##########
                    text_vision_bank_before = []
                    text_vision_bank_before.append((x, src_img_features))

                    if idx == 0:

                        x, src_img_features = layer(x.transpose(0, 1), encoder_padding_mask_text,
                                                    encoder_padding_mask_hallucination, batch_len,
                                                    idx,
                                                    x_change_after=None,
                                                    src_img_features=src_img_features)
                        all_src_img_features.append(src_img_features)

                        voting_score = []
                        feature_change = []

                        gate_text, gate_vision = self.BalancedFeatureSelector(x, src_img_features)

                        for i in range(0, 3):

                            text_idx, vision_idx = self.probability_sampling(gate_text.transpose(0, 1), gate_vision, 2,
                                                                             src_tokens, 1)

                            ######################modality mixing############################
                            x_change_after, src_img_features = self.exchange_tokens(x.transpose(0, 1), src_img_features,
                                                                                    text_idx, vision_idx)
                            #################################################################

                            ####################voting mechanism ###################
                            kl_change = self.kl_divergence_loss(x.transpose(0, 1), x_change_after)

                            voting_score.append(kl_change)

                            feature_change.append((x_change_after, src_img_features))

                        voting_score_values = [score.item() for score in voting_score]

                        min_score_index = voting_score_values.index(min(voting_score_values))

                        x_change_after, src_img_features = feature_change[min_score_index]
                        ###################################################################


                        ################cross-modal gating################
                        x = self.cross_gating(hallucination, x_change_after.transpose(0, 1)) + hallucination


                        x_oral_representation.append(x)
                        x_change_representation.append(x_change_after)

                    else:

                        x, src_img_features, x_change_after = layer(x.transpose(0, 1),
                                                                    encoder_padding_mask_text,
                                                                    encoder_padding_mask_hallucination, batch_len,
                                                                    idx,
                                                                    x_change_after=x_change_after,
                                                                    src_img_features=src_img_features)

                        all_src_img_features.append(src_img_features)

                        voting_score = []
                        feature_change = []

                        x_change_after = x_change_after.transpose(0, 1)
                        x = x.transpose(0, 1)

                        gate_text, gate_vision = self.BalancedFeatureSelector(x_change_after, src_img_features)

                        for i in range(0, 3):

                            text_idx, vision_idx = self.probability_sampling(gate_text, gate_vision, 2,
                                                                             src_tokens, 1)

                            ######################modality mixing############################
                            x_change_after, src_img_features = self.exchange_tokens(x_change_after, src_img_features,
                                                                                    text_idx, vision_idx)
                            #################################################################

                            ####################voting mechanism ###################
                            kl_change = self.kl_divergence_loss(x, x_change_after)

                            voting_score.append(kl_change)

                            feature_change.append((x_change_after, src_img_features))
                            #######################################################

                        voting_score_values = [score.item() for score in voting_score]

                        min_score_index = voting_score_values.index(min(voting_score_values))

                        x_change_after, src_img_features = feature_change[min_score_index]

                        x_oral_representation.append(x)
                        x_change_representation.append(x_change_after)

                        x = self.cross_gating(hallucination, x_change_after.transpose(0, 1)) + hallucination

                else:

                    if idx == 0:

                        src_img_features = torch.mean(hallucination, dim=0).repeat(49, 1, 1)

                        x, x_change_after = layer(x.transpose(0, 1),
                                                  encoder_padding_mask_text,
                                                  encoder_padding_mask_hallucination, batch_len,
                                                  idx,
                                                  x_change_after=None,
                                                  src_img_features=src_img_features)

                        x = x.transpose(0, 1)
                        src_img_features = src_img_features.transpose(0, 1)

                        voting_score = []
                        feature_change = []

                        gate_text, gate_vision = self.BalancedFeatureSelector(x, src_img_features)

                        for i in range(0, 3):

                            text_idx, vision_idx = self.probability_sampling(gate_text, gate_vision, 2,
                                                                             src_tokens, 1)

                            ######################modality mixing############################
                            x_change_after, src_img_features = self.exchange_tokens(x, src_img_features,
                                                                                    text_idx, vision_idx)
                            #################################################################

                            ####################voting mechanism ###################
                            kl_change = self.kl_divergence_loss(x, x_change_after)

                            voting_score.append(kl_change)

                            feature_change.append((x_change_after, src_img_features))
                            #######################################################

                        voting_score_values = [score.item() for score in voting_score]

                        min_score_index = voting_score_values.index(min(voting_score_values))

                        x_change_after, src_img_features = feature_change[min_score_index]

                        x_oral_representation.append(x)
                        x_change_representation.append(x_change_after)

                        x = self.cross_gating( hallucination.transpose(0,1), x_change_after) + hallucination.transpose(0,1)

                        src_img_features = None

                    else:


                        src_img_features = torch.mean(hallucination, dim=0).repeat(49, 1, 1)

                        x, _, x_change_after = layer(x.transpose(0, 1),
                                                     encoder_padding_mask_text,
                                                     encoder_padding_mask_hallucination,
                                                     batch_len,
                                                     idx,
                                                     x_change_after=x_change_after)

                        voting_score = []
                        feature_change = []
                        x = x.transpose(0, 1)
                        src_img_features = src_img_features.transpose(0, 1)
                        x_change_after = x_change_after.transpose(0, 1)

                        gate_text, gate_vision = self.BalancedFeatureSelector(x_change_after, src_img_features)

                        for i in range(0, 3):

                            text_idx, vision_idx = self.probability_sampling(gate_text, gate_vision,
                                                                             2,
                                                                             src_tokens, 1)

                            ######################modality mixing############################
                            x_change_after, src_img_features = self.exchange_tokens(x_change_after,
                                                                                    src_img_features,
                                                                                    text_idx, vision_idx)
                            #################################################################

                            ####################voting mechanism ###################
                            kl_change = self.kl_divergence_loss(x, x_change_after)

                            voting_score.append(kl_change)

                            feature_change.append((x_change_after, src_img_features))
                            #######################################################

                        voting_score_values = [score.item() for score in voting_score]

                        min_score_index = voting_score_values.index(min(voting_score_values))

                        x_change_after, src_img_features = feature_change[min_score_index]

                        x_oral_representation.append(x)
                        x_change_representation.append(x_change_after)

                        x = self.cross_gating(hallucination.transpose(0, 1), x_change_after) + hallucination.transpose(0,1)

                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        if src_img_features is not None:
            hallucination_train = torch.mean((hallucination), dim=0, keepdim=True).repeat(49, 1, 1)

            hallucination_test = hallucination

        else:
            hallucination_test = x_change_after

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        if src_img_features is not None:
            return EncoderOut(
                encoder_out=x,  # T x B x C
                encoder_padding_mask=encoder_padding_mask_text,  # B x T
                encoder_embedding=encoder_embedding,  # B x T x C
                encoder_states=encoder_states,  # List[T x B x C]
                hallucination_train=hallucination_train.transpose(0, 1),
                hallucination_test=hallucination_test.transpose(0, 1),
                src_img_features=all_src_img_features
            )
        else:
            return EncoderOut(
                encoder_out=x,  # T x B x C
                encoder_padding_mask=encoder_padding_mask_text,  # B x T
                encoder_embedding=encoder_embedding,  # B x T x C
                encoder_states=encoder_states,  # List[T x B x C]
                hallucination_train=None,
                hallucination_test=hallucination_test,
                src_img_features=None
            )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """

        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )

        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(
                    0, new_order
                )
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(
                    0, new_order
                )
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, "_future_mask")
                or self._future_mask is None
                or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(
                    utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
                )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.args.encoder_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        self.adaptive_softmax = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(
                args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        # print("encoder_out.encoder_padding_mask:", encoder_out.encoder_padding_mask)
        for idx, layer in enumerate(self.layers):
            encoder_state: Optional[Tensor] = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_states = encoder_out.encoder_states
                    assert encoder_states is not None
                    encoder_state = encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn, _ = layer(
                    x,
                    encoder_out.encoder_out,
                    encoder_out.encoder_padding_mask
                    if encoder_out is not None
                    else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        # return x, {"attn": [attn], "inner_states": inner_states, "txt_out": encoder_out.txt_out, "img_out": encoder_out.img_out}

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    m.weight = nn.Parameter(m.weight.half())
    if bias:
        m.bias = nn.Parameter(m.bias.half())
    return m.to(torch.float16)


@register_model_architecture("transformer", "transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)


@register_model_architecture("transformer", "transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.gating_dim = getattr(args, "gating_dim", 256)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer", "transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer", "transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer", "transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer_align", "transformer_align")
def transformer_align(args):
    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    args.full_context_alignment = getattr(args, "full_context_alignment", False)
    base_architecture(args)


@register_model_architecture("transformer_align", "transformer_wmt_en_de_big_align")
def transformer_wmt_en_de_big_align(args):
    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    transformer_wmt_en_de_big(args)


######### gating  ##########
class GatingMechanism(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fc_img = Linear(args.gating_dim * 2, 1)

    def forward(self, x, grid_img_features):
        grid_img_features = torch.mean(grid_img_features, dim=0, keepdim=True)  ## 1*batch*dim
        t, b, c = x.shape
        grid_img_features = grid_img_features.expand(t, b, c)
        merge = torch.cat([x, grid_img_features], dim=-1)

        gate = torch.sigmoid(self.fc_img(merge))  # T B C
        img_features = torch.mul(gate, x)
        return img_features


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.1):
        super(MLP, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout_prob)
        )
        self.half()

    def forward(self, y):
        y = y.half()
        x = self.fc(y)
        return x


class SelectiveGating_text(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1):
        super(SelectiveGating_text, self).__init__()
        # self.U = nn.Parameter(torch.randn(input_dim, input_dim))
        # self.bias = nn.Parameter(torch.zeros(input_dim))
        self.score_layer = nn.Linear(input_dim, 1)
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, general_rep):
        gate_signal = torch.sigmoid(self.score_layer(general_rep))


        return gate_signal


class SelectiveGating_vision(nn.Module):
    def __init__(self, feature_dim):
        super(SelectiveGating_vision, self).__init__()
        self.score_layer = nn.Linear(feature_dim, 1)

        self.half()

    def forward(self, vit_features):
        scores = F.sigmoid(self.score_layer(vit_features))
        # gated_features = vit_features + scores * vit_features

        # return gated_features, scores
        return scores


class BalancedFeatureSelector(nn.Module):
    def __init__(self, text_input_dim, vision_input_dim):
        super(BalancedFeatureSelector, self).__init__()
        self.text_gate = SelectiveGating_text(text_input_dim)
        self.vision_gate = SelectiveGating_vision(vision_input_dim)

    def forward(self, text_rep, vision_rep):
        # gated_text_rep, gate_text = self.text_gate(text_rep)
        # gated_vision_rep, gate_vision = self.vision_gate(vision_rep)

        gate_text = self.text_gate(text_rep)
        gate_vision = self.vision_gate(vision_rep)

        # return gated_text_rep, gated_vision_rep, gate_text, gate_vision
        return gate_text, gate_vision


class FeatureMapping_vision(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureMapping_vision, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.activation(x)


class FeatureMapping_text(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureMapping_text, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # self.batch_norm = nn.BatchNorm1d(output_dim)
        # self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        # x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        return x


class TextVisualFeatureMapper(nn.Module):
    def __init__(self, text_dim, visual_dim, common_dim):
        super(TextVisualFeatureMapper, self).__init__()
        self.text_mapper = FeatureMapping_text(text_dim, common_dim)
        self.visual_mapper = FeatureMapping_vision(visual_dim, common_dim)

    def forward(self, text_features, visual_features):
        mapped_text_features = self.text_mapper(text_features)
        mapped_visual_features = self.visual_mapper(visual_features)
        return mapped_text_features, mapped_visual_features


class TextSelfAttention(nn.Module):
    def __init__(self, dim):
        super(TextSelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x):
        # x shape: [batch_size, length, dim]

        query = self.query(x)  # [batch_size, length, dim]
        key = self.key(x)  # [batch_size, length, dim]
        value = self.value(x)  # [batch_size, length, dim]

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (
                    self.dim ** 0.5)  # [batch_size, length, length]
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended = torch.matmul(attention_weights, value)  # [batch_size, length, dim]
        return attended


class VisualSelfAttention(nn.Module):
    def __init__(self, dim):
        super(VisualSelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x):
        # x shape: [batch_size, 49, dim]

        query = self.query(x)  # [batch_size, 49, dim]
        key = self.key(x)  # [batch_size, 49, dim]
        value = self.value(x)  # [batch_size, 49, dim]

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.dim ** 0.5)  # [batch_size, 49, 49]
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended = torch.matmul(attention_weights, value)  # [batch_size, 49, dim]
        return attended


class VisualBatchNorm(nn.Module):
    def __init__(self, dim):
        super(VisualBatchNorm, self).__init__()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        batch_size, _, dim = x.size()
        #  [batch_size, 49, dim] -> [batch_size * 49, dim]
        x = x.view(-1, dim)
        x = self.norm(x)
        #  [batch_size * 49, dim] -> [batch_size, 49, dim]
        x = x.view(batch_size, -1, dim)
        return x


class GatingMechanism(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.x_linear = Linear(, 1)
        # self.x_linear = Linear(batch_len, 1)

        self.fc_img = Linear(args.gating_dim * 2, 1)

        # self.fc_img_x = Linear(args.gating_dim, 128)

    def forward(self, x, grid_img_features):
        # x = torch.mean(x, dim=0, keepdim=True)
        # region_x_features = torch.cat([region_img_features, x.repeat(region_img_features.size(0), 1, 1)], dim=-1)
        # region_linear_x = self.fc_img(region_x_features)
        #
        # region_sigmoid_x = torch.sigmoid(region_linear_x)  # max_len * batch * 1
        # region_img_features = torch.mul(region_sigmoid_x, region_img_features)
        # return region_img_features, region_sigmoid_x

        if grid_img_features.size(0) == x.size(0):
            final_representation = torch.cat((grid_img_features, x), dim=-1)
            gate = torch.sigmoid(self.fc_img(final_representation))
            final_representation = torch.mul(gate, x)

        else:
            grid_img_features = torch.mean(grid_img_features.transpose(0, 1), dim=1, keepdim=True)  ## 1*batch*dim
            t, b, c = x.shape
            grid_img_features = grid_img_features.expand(b, t, c)
            merge = torch.cat([x.transpose(0, 1), grid_img_features], dim=-1)
            gate = torch.sigmoid(self.fc_img(merge))  # T B C
            final_representation = torch.mul(gate, x.transpose(0, 1))

        return final_representation
