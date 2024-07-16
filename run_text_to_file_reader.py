import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts(model_id, sentence, filename, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename, duration_scaling_factor=duration_scaling_factor)
    del tts


def the_raven(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=['Once upon a midnight dreary, while I pondered, weak, and weary,',
                         'Over many a quaint, and curious volume, of forgotten lore,',
                         'While I nodded, nearly napping, suddenly, there came a tapping,',
                         'As of someone gently rapping, rapping at my chamber door.',
                         'Ah, distinctly, I remember, it was in the bleak December,',
                         'And each separate dying ember, wrought its ghost upon the floor.',
                         'Eagerly, I wished the morrow, vainly, I had sought to borrow',
                         'From my books surcease of sorrow, sorrow, for the lost Lenore,',
                         'And the silken, sad, uncertain, rustling of each purple curtain',
                         'Thrilled me, filled me, with fantastic terrors, never felt before.'],
               filename=f"audios/{version}_the_raven.wav",
               device=exec_device,
               language="eng",
               speaker_reference=speaker_reference)


def sound_of_silence_single_utt(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["In restless dreams I walked alone, Narrow streets of cobblestone. Beneath the halo of a streetlamp, I turned my collar to the cold and damp,  When my eyes were stabbed, by the flash of a neon light, That split the night. And touched the sound, of silence."],
               filename=f"audios/{version}_sound_of_silence_as_single_utterance.wav",
               device=exec_device,
               language="eng",
               speaker_reference=speaker_reference)


def die_glocke(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""Fest gemauert in der Erden,
                         Steht die Form, aus Lehm gebrannt.
                         Heute muss die Glocke werden!
                         Frisch, Gesellen, seid zur Hand!"""],
               filename=f"audios/{version}_die_glocke.wav",
               device=exec_device,
               language="deu",
               speaker_reference=speaker_reference)


def viet_poem(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""Thân phận,
                            ở một nơi luôn phải nhắc mình,
                            im miệng,
                            thân phận,
                            là khi nói về quá khứ,
                            ngó trước nhìn sau,
                            là phải biết nhắm mắt bịt tai làm lơ,
                            thờ ơ,
                            với tất cả những điều gai chướng,
                            thân phận chúng tôi ở đó,
                            những quyển sách chuyền tay nhau như ăn cắp,
                            ngôn luận ư?
                            không có đất cho nghĩa tự do."""],
               filename=f"audios/{version}_viet_poem.wav",
               device=exec_device,
               language="vie",
               speaker_reference=speaker_reference,
               duration_scaling_factor=1.2)

def variance_test(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["It snowed, rained, and hailed the same morning.",
                         "It snowed, rained, and hailed the same morning.",
                         "It snowed, rained, and hailed the same morning.",
                         "It snowed, rained, and hailed the same morning.",],
               filename=f"audios/{version}_variance_test.wav",
               device=exec_device,
               language="eng",
               speaker_reference=speaker_reference)


if __name__ == '__main__':
    gpu_id = 3
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    # merged_speaker_references = ["audios/speaker_references/" + ref for ref in os.listdir("audios/speaker_references/")]
    """
    sound_of_silence_single_utt(version="CFM_PED",
                                model_id="Libri_Prosody/CFM/pitch_energy_duration",
                                exec_device=exec_device)

    die_glocke(version="CFM_PED",
               model_id="Libri_Prosody/CFM/pitch_energy_duration",
               exec_device=exec_device)

    the_raven(version="CFM_PED",
              model_id="Libri_Prosody/CFM/pitch_energy_duration",
              exec_device=exec_device)
    """
    variance_test(version="epd_log_no_drop",
              model_id="Libri_Prosody/CFM/epd_log_no_drop",
              exec_device=device,
              speaker_reference="audios/speaker_reference/100_121669_000013_000000.wav")
