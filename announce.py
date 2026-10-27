"""
SmartRail - Guidance Messages & Audio Announcement Engine
Deterministic safety and operational announcement templates in English and Hindi.
Generates platform display boards and optional voice playback (via gTTS or offline tone fallback).
Strict safety policy: Deterministic templates only, NO LLM hallucination risk.
"""

import os
import io
import wave
import struct
import math
from typing import Dict, Any, Optional, Tuple

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

AUDIO_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Deterministic safety announcement templates
TEMPLATES = {
    "delay": {
        "en": "Attention please! Train number {train_num}, {train_name}, is running behind schedule by approximately {delay_mins} minutes. Expected arrival at {arrival_time}. Inconvenience caused is deeply regretted.",
        "hi": "यात्रीगण कृपया ध्यान दें! गाड़ी संख्या {train_num}, {train_name}, अपने निर्धारित समय से लगभग {delay_mins} मिनट की देरी से चल रही है। इसके {arrival_time} बजे आने की संभावना है। यात्रियों को हुई असुविधा के लिए हमें खेद है।"
    },
    "platform_change": {
        "en": "Attention please! Train number {train_num}, {train_name}, scheduled on Platform {old_platform}, will now arrive on Platform {new_platform}. Passengers are requested to proceed to Platform {new_platform} using the foot overbridge carefully.",
        "hi": "यात्रीगण कृपया ध्यान दें! गाड़ी संख्या {train_num}, {train_name}, जो पहले प्लेटफ़ॉर्म {old_platform} पर आनी थी, अब प्लेटफ़ॉर्म {new_platform} पर आएगी। कृपया ध्यानपूर्वक पैदल उपरिगामी पुल का उपयोग करते हुए प्लेटफ़ॉर्म {new_platform} पर पहुंचें।"
    },
    "crowd_guidance": {
        "en": "Passenger notice for Train {train_num}: General unreserved coaches are located at Zone A (Front) and Zone C (Rear). Zone {target_zone} is significantly less crowded. Passengers are requested to queue at the marked line in Zone {target_zone} for safe boarding.",
        "hi": "गाड़ी संख्या {train_num} के यात्रियों के लिए विशेष सूचना: सामान्य अनारक्षित डिब्बे ज़ोन A (आगे) और ज़ोन C (पीछे) पर स्थित हैं। ज़ोन {target_zone} में काफी कम भीड़ है। सुरक्षित यात्रा हेतु कृपया ज़ोन {target_zone} में चिन्हित रेखा पर कतार लगाएं।"
    },
    "do_not_litter": {
        "en": "Swachh Bharat Railway Mission: Passengers on Platform {platform} are requested not to litter on platforms or railway tracks. Dedicated dustbins are placed every 30 meters. Help us keep your station clean and hygienic.",
        "hi": "स्वच्छ भारत रेलवे अभियान: प्लेटफ़ॉर्म {platform} के सभी यात्रियों से अनुरोध है कि प्लेटफ़ॉर्म या पटरियों पर कचरा न फेंकें। प्रत्येक 30 मीटर पर कूड़ेदान उपलब्ध हैं। स्टेशन को स्वच्छ और सुंदर बनाए रखने में सहयोग करें।"
    },
    "yellow_line": {
        "en": "Safety Warning: Passengers waiting on Platform {platform} {zone}, please stand behind the yellow safety line until the incoming train has come to a complete halt.",
        "hi": "सुरक्षा चेतावनी: प्लेटफ़ॉर्म {platform} {zone} पर प्रतीक्षा कर रहे यात्रीगण, कृपया ट्रेन के पूर्णतः रुकने तक पीली सुरक्षा रेखा के पीछे खड़े रहें।"
    }
}

def generate_coach_guidance_board(
    train_num: str,
    train_name: str,
    platform: str,
    zone_a_risk: str,
    zone_c_risk: str,
    zone_a_count: int,
    zone_c_count: int
) -> Dict[str, Any]:
    """
    Coach-zone guidance shown on platform display board:
    Picks the less crowded general zone (A or C) when the other is High.
    Instructs passengers to queue at the marked line.
    """
    if zone_c_risk == "High" and zone_a_risk != "High":
        target_zone = "Zone A (Front)"
        recommendation = "General coaches: Zone A is less crowded, please queue there at the marked line."
        recommendation_hi = "सामान्य डिब्बे: ज़ोन A में कम भीड़ है, कृपया वहां चिन्हित रेखा पर कतार लगाएं।"
    elif zone_a_risk == "High" and zone_c_risk != "High":
        target_zone = "Zone C (Rear)"
        recommendation = "General coaches: Zone C is less crowded, please queue there at the marked line."
        recommendation_hi = "सामान्य डिब्बे: ज़ोन C में कम भीड़ है, कृपया वहां चिन्हित रेखा पर कतार लगाएं।"
    else:
        # Balanced or both normal
        less_crowded = "Zone A (Front)" if zone_a_count <= zone_c_count else "Zone C (Rear)"
        target_zone = less_crowded
        recommendation = f"General coaches: {less_crowded} has lighter density. Please queue at the marked line."
        recommendation_hi = f"सामान्य डिब्बे: {less_crowded} पर भीड़ सामान्य है। कृपया चिन्हित रेखा पर कतार लगाएं।"

    return {
        "platform": platform,
        "train_number": train_num,
        "train_name": train_name,
        "target_zone": target_zone,
        "display_text_en": recommendation,
        "display_text_hi": recommendation_hi,
        "marked_line_instruction": "Stand behind the yellow line until boarding is called."
    }

def build_announcement(
    ann_type: str,
    params: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Formats deterministic English and Hindi announcements.
    Returns: (text_en, text_hi)
    """
    template = TEMPLATES.get(ann_type, TEMPLATES["delay"])
    try:
        text_en = template["en"].format(**params)
        text_hi = template["hi"].format(**params)
    except KeyError as e:
        text_en = f"Official railway announcement: {ann_type} notice."
        text_hi = f"आधिकारिक रेलवे सूचना: {ann_type} ध्यान दें।"
    return text_en, text_hi

def create_synthetic_offline_chime() -> bytes:
    """
    Generates a 4-tone Indian Railway station chime (Ding-Dong-Ding-Dong)
    in pure python WAV bytes without any external network or dependencies.
    """
    sample_rate = 22050
    notes = [523.25, 659.25, 783.99, 1046.50] # C5, E5, G5, C6
    duration_per_note = 0.28
    
    audio_data = bytearray()
    for freq in notes:
        n_samples = int(sample_rate * duration_per_note)
        for i in range(n_samples):
            t = i / sample_rate
            # Harmonic decay envelope
            envelope = math.exp(-3.0 * (i / n_samples))
            val = int(32767 * 0.4 * envelope * (
                math.sin(2.0 * math.pi * freq * t) + 
                0.3 * math.sin(4.0 * math.pi * freq * t)
            ))
            val = max(-32767, min(32767, val))
            audio_data.extend(struct.pack('<h', val))

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    
    return buffer.getvalue()

def synthesize_voice(text: str, lang: str = "en") -> Tuple[Optional[bytes], str]:
    """
    Attempts to synthesize voice using gTTS.
    If offline or gTTS fails, falls back safely to railway chime audio bytes
    with a clear status message.
    Returns: (audio_bytes, format_type)
    """
    if GTTS_AVAILABLE:
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read(), "audio/mp3"
        except Exception as e:
            # Clean fallback to offline chime
            print(f"[SmartRail Announce] gTTS offline fallback: {e}")
            return create_synthetic_offline_chime(), "audio/wav"
    else:
        return create_synthetic_offline_chime(), "audio/wav"
