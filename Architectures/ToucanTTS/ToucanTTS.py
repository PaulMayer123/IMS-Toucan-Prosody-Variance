import torch
import torch.nn.functional as torchfunc
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import Tanh

from Architectures.GeneralLayers.ConditionalLayerNorm import AdaIN1d
from Architectures.GeneralLayers.ConditionalLayerNorm import ConditionalLayerNorm
from Architectures.GeneralLayers.Conformer import Conformer
from Architectures.GeneralLayers.LayerNorm import LayerNorm
from Architectures.GeneralLayers.LengthRegulator import LengthRegulator
from Architectures.ToucanTTS.StochasticToucanTTSLoss import StochasticToucanTTSLoss
from Architectures.ToucanTTS.flow_matching import CFMDecoder
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from Utility.utils import initialize
from Utility.utils import integrate_with_utt_embed
from Utility.utils import make_non_pad_mask


class ToucanTTS(torch.nn.Module):
    """
    ToucanTTS module, which is based on a FastSpeech 2 module,
    but with lots of designs from different architectures accumulated
    and some major components added to put a large focus on
    multilinguality and controllability.

    Contributions inspired from elsewhere:
    - The Decoder is a flow matching network, like in Matcha-TTS and StableTTS
    - Pitch and energy values are averaged per-phone, as in FastPitch to enable great controllability
    - The encoder and decoder are Conformers, like in ESPnet

    """

    def __init__(self,
                 # network structure related
                 input_feature_dimensions=64,
                 spec_channels=128,
                 attention_dimension=384,
                 attention_heads=4,
                 positionwise_conv_kernel_size=1,
                 use_scaled_positional_encoding=True,
                 init_type="xavier_uniform",
                 use_macaron_style_in_conformer=True,
                 use_cnn_in_conformer=True,

                 # encoder
                 encoder_layers=6,
                 encoder_units=1536,
                 encoder_normalize_before=True,
                 encoder_concat_after=False,
                 conformer_encoder_kernel_size=7,
                 transformer_enc_dropout_rate=0.1,
                 transformer_enc_positional_dropout_rate=0.1,
                 transformer_enc_attn_dropout_rate=0.1,

                 # decoder
                 decoder_layers=6,
                 decoder_units=1536,
                 decoder_concat_after=False,
                 conformer_decoder_kernel_size=31,  # 31 works for spectrograms
                 decoder_normalize_before=True,
                 transformer_dec_dropout_rate=0.1,
                 transformer_dec_positional_dropout_rate=0.1,
                 transformer_dec_attn_dropout_rate=0.1,

                 duration_log_scale = True,
                 dropout = True,
                 # order of pitch, energy, duration
                 prosody_order="ped",
                
                 # duration predictor
                 prosody_channels=8,
                 duration_predictor_layers=3,
                 duration_predictor_kernel_size=5,
                 duration_predictor_dropout_rate=0.2,

                 # pitch predictor
                 pitch_predictor_layers=3,
                 pitch_predictor_kernel_size=5,
                 pitch_predictor_dropout=0.2,
                 pitch_embed_kernel_size=1,
                 pitch_embed_dropout=0.0,

                 # energy predictor
                 energy_predictor_layers=2,
                 energy_predictor_kernel_size=3,
                 energy_predictor_dropout=0.2,
                 energy_embed_kernel_size=1,
                 energy_embed_dropout=0.0,

                 # cfm decoder
                 cfm_filter_channels=512,
                 cfm_heads=4,
                 cfm_layers=3,
                 cfm_kernel_size=5,
                 cfm_p_dropout=0.1,

                 # additional features
                 utt_embed_dim=192,  # 192 dim speaker embedding + 16 dim prosody embedding optionally (see older version, this one doesn't use the prosody embedding)
                 lang_embs=8000,
                 lang_emb_size=16,  # lower dimensions seem to work better
                 integrate_language_embedding_into_encoder_out=False,
                 embedding_integration="AdaIN",  # ["AdaIN" | "ConditionalLayerNorm" | "ConcatProject"]
                 ):
        super().__init__()

        self.config = {
            "input_feature_dimensions"                     : input_feature_dimensions,
            "attention_dimension"                          : attention_dimension,
            "attention_heads"                              : attention_heads,
            "positionwise_conv_kernel_size"                : positionwise_conv_kernel_size,
            "use_scaled_positional_encoding"               : use_scaled_positional_encoding,
            "init_type"                                    : init_type,
            "use_macaron_style_in_conformer"               : use_macaron_style_in_conformer,
            "use_cnn_in_conformer"                         : use_cnn_in_conformer,
            "encoder_layers"                               : encoder_layers,
            "encoder_units"                                : encoder_units,
            "encoder_normalize_before"                     : encoder_normalize_before,
            "encoder_concat_after"                         : encoder_concat_after,
            "conformer_encoder_kernel_size"                : conformer_encoder_kernel_size,
            "transformer_enc_dropout_rate"                 : transformer_enc_dropout_rate,
            "transformer_enc_positional_dropout_rate"      : transformer_enc_positional_dropout_rate,
            "transformer_enc_attn_dropout_rate"            : transformer_enc_attn_dropout_rate,
            "decoder_layers"                               : decoder_layers,
            "decoder_units"                                : decoder_units,
            "decoder_concat_after"                         : decoder_concat_after,
            "conformer_decoder_kernel_size"                : conformer_decoder_kernel_size,
            "decoder_normalize_before"                     : decoder_normalize_before,
            "transformer_dec_dropout_rate"                 : transformer_dec_dropout_rate,
            "transformer_dec_positional_dropout_rate"      : transformer_dec_positional_dropout_rate,
            "transformer_dec_attn_dropout_rate"            : transformer_dec_attn_dropout_rate,
            "duration_log_scale"                           : duration_log_scale,
            "prosody_order"                                : prosody_order,
            "dropout"                                      : dropout,                 
            "duration_predictor_layers"                    : duration_predictor_layers,
            "duration_predictor_kernel_size"               : duration_predictor_kernel_size,
            "duration_predictor_dropout_rate"              : duration_predictor_dropout_rate,
            "pitch_predictor_layers"                       : pitch_predictor_layers,
            "pitch_predictor_kernel_size"                  : pitch_predictor_kernel_size,
            "pitch_predictor_dropout"                      : pitch_predictor_dropout,
            "pitch_embed_kernel_size"                      : pitch_embed_kernel_size,
            "pitch_embed_dropout"                          : pitch_embed_dropout,
            "energy_predictor_layers"                      : energy_predictor_layers,
            "energy_predictor_kernel_size"                 : energy_predictor_kernel_size,
            "energy_predictor_dropout"                     : energy_predictor_dropout,
            "energy_embed_kernel_size"                     : energy_embed_kernel_size,
            "energy_embed_dropout"                         : energy_embed_dropout,
            "spec_channels"                                : spec_channels,
            "cfm_filter_channels"                          : cfm_filter_channels,
            "prosody_channels"                             : prosody_channels,
            "cfm_heads"                                    : cfm_heads,
            "cfm_layers"                                   : cfm_layers,
            "cfm_kernel_size"                              : cfm_kernel_size,
            "cfm_p_dropout"                                : cfm_p_dropout,
            "utt_embed_dim"                                : utt_embed_dim,
            "lang_embs"                                    : lang_embs,
            "lang_emb_size"                                : lang_emb_size,
            "embedding_integration"                        : embedding_integration,
            "integrate_language_embedding_into_encoder_out": integrate_language_embedding_into_encoder_out
        }

        self.input_feature_dimensions = input_feature_dimensions
        self.attention_dimension = attention_dimension
        self.use_scaled_pos_enc = use_scaled_positional_encoding
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None
        self.integrate_language_embedding_into_encoder_out = integrate_language_embedding_into_encoder_out
        self.use_conditional_layernorm_embedding_integration = embedding_integration in ["AdaIN", "ConditionalLayerNorm"]

        articulatory_feature_embedding = Sequential(Linear(input_feature_dimensions, 100), Tanh(), Linear(100, attention_dimension))
        self.encoder = Conformer(conformer_type="encoder",
                                 attention_dim=attention_dimension,
                                 attention_heads=attention_heads,
                                 linear_units=encoder_units,
                                 num_blocks=encoder_layers,
                                 input_layer=articulatory_feature_embedding,
                                 dropout_rate=transformer_enc_dropout_rate,
                                 positional_dropout_rate=transformer_enc_positional_dropout_rate,
                                 attention_dropout_rate=transformer_enc_attn_dropout_rate,
                                 normalize_before=encoder_normalize_before,
                                 concat_after=encoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                                 macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=True,
                                 cnn_module_kernel=conformer_encoder_kernel_size,
                                 zero_triu=False,
                                 utt_embed=utt_embed_dim,
                                 lang_embs=lang_embs,
                                 lang_emb_size=lang_emb_size,
                                 use_output_norm=True,
                                 embedding_integration=embedding_integration)

        if self.integrate_language_embedding_into_encoder_out:
            self.language_embedding_projection = torch.nn.Linear(lang_emb_size, attention_dimension)
            self.language_emb_norm = LayerNorm(attention_dimension)
            if embedding_integration == "AdaIN":
                self.language_embedding_infusion = AdaIN1d(style_dim=attention_dimension, num_features=attention_dimension)
            elif embedding_integration == "ConditionalLayerNorm":
                self.language_embedding_infusion = ConditionalLayerNorm(speaker_embedding_dim=attention_dimension, hidden_dim=attention_dimension)
            else:
                self.language_embedding_infusion = torch.nn.Linear(attention_dimension + attention_dimension, attention_dimension)

        self.pitch_embed = Sequential(torch.nn.Conv1d(in_channels=1,
                                                    out_channels=attention_dimension,
                                                    kernel_size=pitch_embed_kernel_size,
                                                    padding=(pitch_embed_kernel_size - 1) // 2),
                                    torch.nn.Dropout(pitch_embed_dropout))

        self.energy_embed = Sequential(torch.nn.Conv1d(in_channels=1, out_channels=attention_dimension, kernel_size=energy_embed_kernel_size,
                                                    padding=(energy_embed_kernel_size - 1) // 2),
                                    torch.nn.Dropout(energy_embed_dropout))

        self.length_regulator = LengthRegulator()

        self.decoder = Conformer(conformer_type="decoder",
                                 attention_dim=attention_dimension,
                                 attention_heads=attention_heads,
                                 linear_units=decoder_units,
                                 num_blocks=decoder_layers,
                                 input_layer=None,
                                 dropout_rate=transformer_dec_dropout_rate,
                                 positional_dropout_rate=transformer_dec_positional_dropout_rate,
                                 attention_dropout_rate=transformer_dec_attn_dropout_rate,
                                 normalize_before=decoder_normalize_before,
                                 concat_after=decoder_concat_after,
                                 positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                                 macaron_style=use_macaron_style_in_conformer,
                                 use_cnn_module=use_cnn_in_conformer,
                                 cnn_module_kernel=conformer_decoder_kernel_size,
                                 use_output_norm=embedding_integration not in ["AdaIN", "ConditionalLayerNorm"],
                                 utt_embed=utt_embed_dim,
                                 embedding_integration=embedding_integration)

        self.output_projection = torch.nn.Linear(attention_dimension, spec_channels)
        self.cfm_projection = torch.nn.Linear(attention_dimension, spec_channels)
        if prosody_order != "all":
            self.pitch_latent_reduction = torch.nn.Linear(attention_dimension, prosody_channels)
            self.energy_latent_reduction = torch.nn.Linear(attention_dimension, prosody_channels)
            self.duration_latent_reduction = torch.nn.Linear(attention_dimension, prosody_channels)
        else:
            self.prosody_latent_reduction = torch.nn.Linear(attention_dimension, prosody_channels)

        # initialize parameters
        self._reset_parameters(init_type=init_type)
        if lang_embs is not None:
            torch.nn.init.normal_(self.encoder.language_embedding.weight, mean=0, std=attention_dimension ** -0.5)


        self.duration_log_scale = duration_log_scale
        self.prosody_order = prosody_order
        self.dropout = dropout

        # the following modules have their own init function, so they come AFTER the init.
        if prosody_order != "all":
            self.duration_predictor = CFMDecoder(hidden_channels=prosody_channels,
                                                out_channels=1,
                                                filter_channels=prosody_channels,
                                                n_heads=1,
                                                n_layers=duration_predictor_layers,
                                                kernel_size=duration_predictor_kernel_size,
                                                p_dropout=duration_predictor_dropout_rate,
                                                gin_channels=utt_embed_dim)

            self.pitch_predictor = CFMDecoder(hidden_channels=prosody_channels,
                                            out_channels=1,
                                            filter_channels=prosody_channels,
                                            n_heads=1,
                                            n_layers=pitch_predictor_layers,
                                            kernel_size=pitch_predictor_kernel_size,
                                            p_dropout=pitch_predictor_dropout,
                                            gin_channels=utt_embed_dim)

            self.energy_predictor = CFMDecoder(hidden_channels=prosody_channels,
                                            out_channels=1,
                                            filter_channels=prosody_channels,
                                            n_heads=1,
                                            n_layers=energy_predictor_layers,
                                            kernel_size=energy_predictor_kernel_size,
                                            p_dropout=energy_predictor_dropout,
                                            gin_channels=utt_embed_dim)
        else:
            self.prosody_predictor = CFMDecoder(hidden_channels=prosody_channels,
                                            out_channels=4,
                                            filter_channels=prosody_channels,
                                            n_heads=1,
                                            n_layers=8,
                                            kernel_size=5,
                                            p_dropout=0.2,
                                            gin_channels=utt_embed_dim)
            


        self.flow_matching_decoder = CFMDecoder(hidden_channels=spec_channels, # 128
                                                out_channels=spec_channels, # 128
                                                filter_channels=cfm_filter_channels, # 512
                                                n_heads=cfm_heads, # 4
                                                n_layers=cfm_layers, # 4
                                                kernel_size=cfm_kernel_size, # 5
                                                p_dropout=cfm_p_dropout, # 0.1
                                                gin_channels=utt_embed_dim)
        
              

        self.criterion = StochasticToucanTTSLoss()

    def forward(self,
                text_tensors,
                text_lengths,
                gold_speech,
                speech_lengths,
                gold_durations,
                gold_pitch,
                gold_energy,
                utterance_embedding,
                return_feats=False,
                lang_ids=None,
                run_stochastic=True
                ):
        """
        Args:
            return_feats (Boolean): whether to return the predicted spectrogram
            text_tensors (LongTensor): Batch of padded text vectors (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            gold_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_durations (LongTensor): Batch of padded durations (B, Tmax + 1).
            gold_pitch (Tensor): Batch of padded token-averaged pitch (B, Tmax + 1, 1).
            gold_energy (Tensor): Batch of padded token-averaged energy (B, Tmax + 1, 1).
            lang_ids (LongTensor): The language IDs used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Batch of embeddings to condition the TTS on, if the model is multispeaker
            run_stochastic (Bool): Whether to detach the inputs to the normalizing flow for stability.
        """
        outs, \
            stochastic_loss, \
            duration_loss, \
            pitch_loss, \
            energy_loss = self._forward(text_tensors=text_tensors,
                                        text_lengths=text_lengths,
                                        gold_speech=gold_speech,
                                        speech_lengths=speech_lengths,
                                        gold_durations=gold_durations,
                                        gold_pitch=gold_pitch,
                                        gold_energy=gold_energy,
                                        utterance_embedding=utterance_embedding,
                                        is_inference=False,
                                        lang_ids=lang_ids,
                                        run_stochastic=run_stochastic)

        # calculate loss
        regression_loss = self.criterion(predicted_features=outs,
                                         gold_features=gold_speech,
                                         features_lengths=speech_lengths)

        if pitch_loss != None: # if pitch is None only duration loss is whole prosody loss
            if return_feats:
                return regression_loss, stochastic_loss, duration_loss, pitch_loss, energy_loss, outs
            return regression_loss, stochastic_loss, duration_loss, pitch_loss, energy_loss
        else:    
            if return_feats:
                return regression_loss, stochastic_loss, duration_loss, outs
            return regression_loss, stochastic_loss, duration_loss

    def _forward(self,
                 text_tensors,
                 text_lengths,
                 gold_speech=None,
                 speech_lengths=None,
                 gold_durations=None,
                 gold_pitch=None,
                 gold_energy=None,
                 is_inference=False,
                 utterance_embedding=None,
                 lang_ids=None,
                 run_stochastic=False):

        text_tensors = torch.clamp(text_tensors, max=1.0)
        # this is necessary, because of the way we represent modifiers to keep them identifiable.

        utterance_embedding = torch.nn.functional.normalize(utterance_embedding)

        if not self.multilingual_model:
            lang_ids = None

        if not self.multispeaker_model:
            utterance_embedding = None

        # encoding the texts
        text_masks = make_non_pad_mask(text_lengths, device=text_lengths.device).unsqueeze(-2)
        encoded_texts, _ = self.encoder(text_tensors, text_masks, utterance_embedding=utterance_embedding, lang_ids=lang_ids)

        if self.integrate_language_embedding_into_encoder_out:
            lang_embs = self.encoder.language_embedding(lang_ids)
            lang_embs = self.language_embedding_projection(lang_embs)
            lang_embs = self.language_emb_norm(lang_embs)
            encoded_texts = integrate_with_utt_embed(hs=encoded_texts, utt_embeddings=lang_embs, projection=self.language_embedding_infusion, embedding_training=self.use_conditional_layernorm_embedding_integration)

        
        if is_inference:
            # predicting pitch, energy and durations
            if self.prosody_order != "all":
                input_energy = encoded_texts

                if self.prosody_order=="ped":
                    
                    reduced_pitch_space = self.pitch_latent_reduction(encoded_texts)
                    if self.dropout:
                        reduced_pitch_space = torchfunc.dropout(reduced_pitch_space, p=0.2)
                    reduced_pitch_space = reduced_pitch_space.transpose(1, 2)
                    
                    pitch_predictions = self.pitch_predictor(mu=reduced_pitch_space, mask=text_masks.float(), n_timesteps=10, temperature=1.0, c=utterance_embedding)
                    embedded_pitch_curve = self.pitch_embed(pitch_predictions).transpose(1, 2)
                    input_energy= encoded_texts + embedded_pitch_curve

                
                reduced_energy_space = self.energy_latent_reduction(input_energy)
                if self.dropout:
                    reduced_energy_space = torchfunc.dropout(reduced_energy_space, p=0.2)
                reduced_energy_space = reduced_energy_space.transpose(1, 2)
                
                energy_predictions = self.energy_predictor(mu=reduced_energy_space, mask=text_masks.float(), n_timesteps=10, temperature=1.0, c=utterance_embedding)
                embedded_energy_curve = self.energy_embed(energy_predictions).transpose(1, 2)

                if self.prosody_order=="epd":
                    reduced_pitch_space = self.pitch_latent_reduction(encoded_texts + embedded_energy_curve)
                    if self.dropout:
                        reduced_pitch_space = torchfunc.dropout(reduced_pitch_space, p=0.2)
                    reduced_pitch_space = reduced_pitch_space.transpose(1, 2)
                    pitch_predictions = self.pitch_predictor(mu=reduced_pitch_space, mask=text_masks.float(), n_timesteps=10, temperature=1.0, c=utterance_embedding)
                    embedded_pitch_curve = self.pitch_embed(pitch_predictions).transpose(1, 2)

                reduced_duration_space = self.duration_latent_reduction(encoded_texts + embedded_pitch_curve + embedded_energy_curve)
                if self.dropout:
                    reduced_duration_space = torchfunc.dropout(reduced_duration_space, p=0.2)
                reduced_duration_space = reduced_duration_space.transpose(1, 2)
                predicted_durations = self.duration_predictor(mu=reduced_duration_space, mask=text_masks.float(), n_timesteps=10, temperature=1.0, c=utterance_embedding)
                
                
            else:
               
                reduced_prosody_space = self.prosody_latent_reduction(encoded_texts)
                if self.dropout:
                    reduced_prosody_space = torchfunc.dropout(reduced_prosody_space, p=0.2)
                reduced_prosody_space = reduced_prosody_space.transpose(1, 2)
                predicted_prosody = self.prosody_predictor(mu=reduced_prosody_space, mask=text_masks.float(), n_timesteps=30, temperature=1.0, c=utterance_embedding)
                
                predicted_durations = predicted_prosody[:, 0:1, :]
                pitch_predictions = predicted_prosody[:, 1:2, :]
                embedded_pitch_curve = self.pitch_embed(pitch_predictions).transpose(1, 2)
                energy_predictions = predicted_prosody[:, 2:3, :]
                embedded_energy_curve = self.energy_embed(energy_predictions).transpose(1, 2)
            
            # change value range 
            #print("predicted: ", predicted_durations)
            #predicted_durations = predicted_durations * 50
            #print("predicted2: ", predicted_durations)

            # convert from log
            if self.duration_log_scale:
                predicted_durations = torch.clamp(predicted_durations, min=0.0).squeeze(1)
                predicted_durations = torch.exp(predicted_durations) - 1
                predicted_durations = predicted_durations.int()
            else:
                predicted_durations = torch.clamp(torch.ceil(predicted_durations), min=0.0).long().squeeze(1)


            for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze(0)):
                if phoneme_vector[get_feature_to_index_lookup()["word-boundary"]] == 1:
                    predicted_durations[0][phoneme_index] = 0

            # enriching the text with pitch and energy info
            enriched_encoded_texts = encoded_texts + embedded_pitch_curve + embedded_energy_curve

            # predicting durations for text and upsampling accordingly
            upsampled_enriched_encoded_texts = self.length_regulator(enriched_encoded_texts, predicted_durations)

        else:
            # training with teacher forcing
            # changing value range
            #shifted_gold = gold_durations / 50
            if self.duration_log_scale:
                shifted_gold = gold_durations + 1
                transformed_gold_durations = torch.log(shifted_gold)
                transformed_gold_durations = transformed_gold_durations.unsqueeze(-1).transpose(1, 2)
            else:
                transformed_gold_durations = gold_durations.unsqueeze(-1).transpose(1, 2).float()

            if self.prosody_order != "all":
                input_energy = encoded_texts
                if self.prosody_order == "ped":
                    reduced_pitch_space = self.pitch_latent_reduction(encoded_texts)
                    if self.dropout:
                        reduced_pitch_space = torchfunc.dropout(reduced_pitch_space, p=0.2)
                    reduced_pitch_space = reduced_pitch_space.transpose(1, 2)
                    pitch_loss, _ = self.pitch_predictor.compute_loss(mu=reduced_pitch_space,
                                                                    x1=gold_pitch.transpose(1, 2),
                                                                    mask=text_masks.float(),
                                                                    c=utterance_embedding)
                    embedded_pitch_curve = self.pitch_embed(gold_pitch.transpose(1, 2)).transpose(1, 2)
                    input_energy = encoded_texts + embedded_pitch_curve

            
                reduced_energy_space = self.energy_latent_reduction(input_energy)
                if self.dropout:
                    reduced_energy_space = torchfunc.dropout(reduced_energy_space, p=0.2)
                reduced_energy_space = reduced_energy_space.transpose(1, 2)
                energy_loss, _ = self.energy_predictor.compute_loss(mu=reduced_energy_space,
                                                                    x1=gold_energy.transpose(1, 2),
                                                                    mask=text_masks.float(),
                                                                    c=utterance_embedding)
                embedded_energy_curve = self.energy_embed(gold_energy.transpose(1, 2)).transpose(1, 2)  
                
                if self.prosody_order == "epd":
                    reduced_pitch_space = self.pitch_latent_reduction(encoded_texts+ embedded_energy_curve)
                    if self.dropout:
                        reduced_pitch_space = torchfunc.dropout(reduced_pitch_space, p=0.2)
                    reduced_pitch_space = reduced_pitch_space.transpose(1, 2)
                    pitch_loss, _ = self.pitch_predictor.compute_loss(mu=reduced_pitch_space,
                                                                    x1=gold_pitch.transpose(1, 2),
                                                                    mask=text_masks.float(),
                                                                    c=utterance_embedding)
                    embedded_pitch_curve = self.pitch_embed(gold_pitch.transpose(1, 2)).transpose(1, 2)
                
                reduced_duration_space = self.duration_latent_reduction(encoded_texts + embedded_pitch_curve + embedded_energy_curve)
                if self.dropout:
                    reduced_duration_space = torchfunc.dropout(reduced_duration_space, p=0.2)
                reduced_duration_space = reduced_duration_space.transpose(1, 2)
                duration_loss, _ = self.duration_predictor.compute_loss(mu=reduced_duration_space,
                                                                        x1=transformed_gold_durations,
                                                                        mask=text_masks.float(),
                                                                        c=utterance_embedding)
                
            else:
                reduced_prosody_space = self.prosody_latent_reduction(encoded_texts)
                if self.dropout:
                    reduced_prosody_space = torchfunc.dropout(reduced_prosody_space, p=0.2)
                reduced_prosody_space = reduced_prosody_space.transpose(1, 2)
                

                prosody_loss, _ = self.prosody_predictor.compute_loss(mu=reduced_prosody_space,
                                                       x1=torch.cat((
                                                           transformed_gold_durations,
                                                           gold_pitch.transpose(1, 2),
                                                           gold_energy.transpose(1, 2),
                                                           torch.zeros_like(gold_energy.transpose(1, 2))
                                                       ), dim=1),
                                                       mask=text_masks.float(),
                                                       c=utterance_embedding)
                
                embedded_energy_curve = self.energy_embed(gold_energy.transpose(1, 2)).transpose(1, 2)  
                embedded_pitch_curve = self.pitch_embed(gold_pitch.transpose(1, 2)).transpose(1, 2)
                duration_loss = prosody_loss # setting duration loss as whole prosody loss
                pitch_loss = None
                energy_loss = None

            enriched_encoded_texts = encoded_texts + embedded_pitch_curve + embedded_energy_curve
            
            upsampled_enriched_encoded_texts = self.length_regulator(enriched_encoded_texts, gold_durations)

        # decoding spectrogram
        decoder_masks = make_non_pad_mask(speech_lengths, device=speech_lengths.device).unsqueeze(-2) if speech_lengths is not None and not is_inference else None
        decoded_speech, _ = self.decoder(upsampled_enriched_encoded_texts, decoder_masks, utterance_embedding=utterance_embedding)

        preliminary_spectrogram = self.output_projection(decoded_speech)

        if is_inference:
            if run_stochastic:
                refined_codec_frames = self.flow_matching_decoder(mu=self.cfm_projection(decoded_speech).transpose(1, 2),
                                                                  mask=make_non_pad_mask([len(decoded_speech[0])], device=decoded_speech.device).unsqueeze(-2).float(),
                                                                  n_timesteps=15,
                                                                  temperature=0.2,
                                                                  c=utterance_embedding).transpose(1, 2)
            else:
                refined_codec_frames = preliminary_spectrogram
            return refined_codec_frames, \
                predicted_durations.squeeze(), \
                pitch_predictions.squeeze(), \
                energy_predictions.squeeze()
        else:
            if run_stochastic:
                stochastic_loss, _ = self.flow_matching_decoder.compute_loss(x1=gold_speech.transpose(1, 2),
                                                                             mask=decoder_masks.float(),
                                                                             mu=self.cfm_projection(decoded_speech).transpose(1, 2),
                                                                             c=utterance_embedding)
            else:
                stochastic_loss = None
            return preliminary_spectrogram, \
                stochastic_loss, \
                duration_loss, \
                pitch_loss, \
                energy_loss

    @torch.inference_mode()
    def inference(self,
                  text,
                  speech=None,
                  utterance_embedding=None,
                  return_duration_pitch_energy=False,
                  lang_id=None,
                  run_stochastic=True):
        """
        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            return_duration_pitch_energy (Boolean): whether to return the list of predicted durations for nicer plotting
            lang_id (LongTensor): The language ID used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Embedding to condition the TTS on, if the model is multispeaker
            run_stochastic (bool): whether to use the output of the stochastic or of the out_projection to generate codec frames
        """
        self.eval()

        # setup batch axis
        ilens = torch.tensor([text.shape[0]], dtype=torch.long, device=text.device)
        text_pseudobatched, speech_pseudobatched = text.unsqueeze(0), None
        if speech is not None:
            speech_pseudobatched = speech.unsqueeze(0)
        utterance_embeddings = utterance_embedding.unsqueeze(0) if utterance_embedding is not None else None

        outs, \
            duration_predictions, \
            pitch_predictions, \
            energy_predictions = self._forward(text_pseudobatched,
                                               ilens,
                                               speech_pseudobatched,
                                               is_inference=True,
                                               utterance_embedding=utterance_embeddings,
                                               lang_ids=lang_id,
                                               run_stochastic=run_stochastic)  # (1, L, odim)
        self.train()

        if return_duration_pitch_energy:
            return outs.squeeze().transpose(0, 1), duration_predictions, pitch_predictions, energy_predictions
        return outs.squeeze().transpose(0, 1)

    def _reset_parameters(self, init_type="xavier_uniform"):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

    def reset_postnet(self, init_type="xavier_uniform"):
        # useful for after they explode
        initialize(self.flow_matching_decoder, init_type)


if __name__ == '__main__':
    model = ToucanTTS()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(" TESTING TRAINING ")

    dummy_text_batch = torch.randint(low=0, high=2, size=[3, 3, 64]).float()  # [Batch, Sequence Length, Features per Phone]
    dummy_text_lens = torch.LongTensor([2, 3, 3])

    dummy_speech_batch = torch.randn([3, 30, 128])  # [Batch, Sequence Length, Spectrogram Buckets]
    dummy_speech_lens = torch.LongTensor([10, 30, 20])

    dummy_durations = torch.LongTensor([[10, 0, 0], [10, 15, 5], [5, 5, 10]])
    dummy_pitch = torch.Tensor([[[1.0], [0.], [0.]], [[1.1], [1.2], [0.8]], [[1.1], [1.2], [0.8]]])
    dummy_energy = torch.Tensor([[[1.0], [1.3], [0.]], [[1.1], [1.4], [0.8]], [[1.1], [1.2], [0.8]]])

    dummy_utterance_embed = torch.randn([3, 192])  # [Batch, Dimensions of Speaker Embedding]
    dummy_language_id = torch.LongTensor([5, 3, 2])

    ce, fl, dl, pl, el = model(dummy_text_batch,
                               dummy_text_lens,
                               dummy_speech_batch,
                               dummy_speech_lens,
                               dummy_durations,
                               dummy_pitch,
                               dummy_energy,
                               utterance_embedding=dummy_utterance_embed,
                               lang_ids=dummy_language_id)

    loss = ce + dl + pl + el + fl
    print(loss)
    loss.backward()

    print(" TESTING INFERENCE ")
    dummy_text_batch = torch.randint(low=0, high=2, size=[12, 64]).float()  # [Sequence Length, Features per Phone]
    dummy_utterance_embed = torch.randn([192])  # [Dimensions of Speaker Embedding]
    dummy_language_id = torch.LongTensor([2])
    print(model.inference(dummy_text_batch,
                          utterance_embedding=dummy_utterance_embed,
                          lang_id=dummy_language_id).shape)
