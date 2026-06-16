"""Label mappings from ESC-50 and FSD50K to Broad Sound Taxonomy (BST) classes.

BST Quick Reference:
  fx-o  = Everyday objects, tools, weapons (daily objects, iron, clothes)
  fx-v  = VEHICLES (car, plane, bike, ship — isolated vehicle events)
  fx-m  = Machines EXCEPT vehicles (drill, lawn mower, gear, chainsaw)
  fx-h  = Human body sounds EXCLUDING speech (breath, sneeze, clapping, walking)
  fx-a  = Animals (cat, insect, sheep, growl, purr)
  fx-n  = Natural elements AND EXPLOSIONS (wind gusts, fire, ice cracks, water splash, stones, explosions)
  fx-ex = EXPERIMENTAL (reversed sounds, weird effects, unusual processing) — NOT explosions!
  fx-el = Electronic/designed (sci-fi, laser, whoosh, boink, cartoon, UI, alerts, notifications)
  ss-n  = Nature soundscapes (forest ambiance, seaside, river, farmland)
  ss-i  = Indoor soundscapes (room ambience, room tone, office, factory, bar)
  ss-u  = Urban soundscapes (city ambience, airport outside, busy road)
  ss-s  = Synthetic soundscapes (artificially-created, imaginary places)
  sp-s  = Solo speech (single voice speaking)
  sp-c  = Conversation/Crowd (several people TALKING, dialogue, playground)
  sp-p  = Processed speech (phone/radio, robotic voice, TTS)
  m-sp  = Solo percussion music (drum passage, drum loop)
  m-si  = Solo instrument music (isolated melody, solo singing, chords from one instrument)
  m-m   = Multiple instruments music (orchestra, band, duet)
  is-p  = Percussion SINGLE NOTES (drum hit, snare, gong, bell, xylophone)
  is-s  = String SINGLE NOTES (guitar pluck, violin note, harp)
  is-w  = Wind SINGLE NOTES (flute note, trumpet, saxophone)
  is-k  = Keyboard SINGLE NOTES (piano key, organ note)
  is-e  = Electronic SINGLE NOTES (synth note, electronic sample)
"""

ESC50_TO_BST = {
    # ===== ANIMALS → fx-a =====
    "dog": "fx-a", "rooster": "fx-a", "pig": "fx-a", "cow": "fx-a",
    "frog": "fx-a", "cat": "fx-a", "hen": "fx-a", "insects": "fx-a",
    "crow": "fx-a", "chirping_birds": "fx-a", "sheep": "fx-a",

    # ===== NATURE SOUNDSCAPES → ss-n =====
    "rain": "ss-n", "sea_waves": "ss-n", "crackling_fire": "ss-n",
    "crickets": "ss-n", "water_drops": "ss-n", "wind": "ss-n",
    "thunderstorm": "ss-n",

    # ===== NATURAL ELEMENTS → fx-n (includes explosions) =====
    "pouring_water": "fx-n",
    "fireworks": "fx-n",        # was fx-ex — explosions belong to fx-n, NOT fx-ex

    # ===== VEHICLES → fx-v (isolated vehicle events) =====
    "car_horn": "fx-v",         # was ss-u — horn is a vehicle sound
    "helicopter": "fx-v",       # was ss-u — helicopter is a vehicle
    "train": "fx-v",            # was ss-u — isolated train sound = vehicle event
    "airplane": "fx-v",         # was fx-m — airplane is a vehicle, not a machine

    # ===== URBAN SOUNDSCAPE → ss-u =====
    "siren": "ss-u",            # kept ss-u — siren in urban context

    # ===== PERCUSSION INSTRUMENT → is-p =====
    "church_bells": "is-p",     # was ss-u — bell = percussion instrument (BST is-p)

    # ===== MACHINES → fx-m (machines except vehicles) =====
    "engine": "fx-m", "chainsaw": "fx-m",
    "hand_saw": "fx-m", "washing_machine": "fx-m", "vacuum_cleaner": "fx-m",
    "clock_tick": "fx-m",

    # ===== HUMAN BODY → fx-h (excluding speech) =====
    "laughing": "fx-h", "sneezing": "fx-h", "clapping": "fx-h",
    "breathing": "fx-h", "coughing": "fx-h", "crying_baby": "fx-h",
    "snoring": "fx-h", "drinking_sipping": "fx-h", "footsteps": "fx-h",
    "brushing_teeth": "fx-h",

    # ===== EVERYDAY OBJECTS → fx-o (objects, tools, weapons) =====
    "door_wood_knock": "fx-o", "mouse_click": "fx-o",
    "keyboard_typing": "fx-o", "door_wood_creaks": "fx-o",
    "can_opening": "fx-o",
    "toilet_flush": "fx-o",     # was fx-n — toilet is a home object, not natural process
    "glass_breaking": "fx-o",   # was fx-ex — breaking objects = fx-o, NOT experimental

    # ===== ELECTRONIC/DESIGNED → fx-el =====
    "clock_alarm": "fx-el",
}

FSD50K_TO_BST = {
    # ===== CONVERSATION/CROWD → sp-c (people TALKING) =====
    "Conversation": "sp-c",
    "Chatter": "sp-c",
    "Crowd": "sp-c",             # crowd talking
    "Human_group_actions": "sp-c",  # broad, but closest match for group talking

    # ===== HUMAN BODY SOUNDS → fx-h (NOT conversation) =====
    # Cheering is vocal but NOT conversation — it's human body sound
    "Cheering": "fx-h",         # was sp-c — vocal cheering ≠ conversation
    "Applause": "fx-h",         # clapping = human body sound, NOT conversation (sp-c)

    # ===== MACHINES → fx-m (individual machine sounds, NOT indoor ambiance) =====
    "Microwave_oven": "fx-m",   # machine sound — creates conflict with ss-i to exclude ambiguous samples

    # ===== INDOOR SOUNDSCAPE → ss-i =====
    "Domestic_sounds_and_home_sounds": "ss-i",

    # ===== NATURAL ELEMENTS & EXPLOSIONS → fx-n =====
    # BST fx-n explicitly includes "explosions" — these are NOT experimental (fx-ex)
    "Explosion": "fx-n",        # was fx-ex — explosions = fx-n per BST
    "Fireworks": "fx-n",        # was fx-ex
    "Boom": "fx-n",             # was fx-ex — explosion-like
    "Crack": "fx-n",            # was fx-ex — ice cracks, wood cracks = natural

    # ===== OBJECTS, TOOLS, WEAPONS → fx-o =====
    # BST fx-o explicitly lists "tools, weapons" — gunshots/impacts go here
    "Gunshot_and_gunfire": "fx-o",  # was fx-ex — weapons listed under fx-o
    "Shatter": "fx-o",          # was fx-ex — breaking objects
    "Slam": "fx-o",             # was fx-ex — slamming objects (doors)
    "Thump_and_thud": "fx-o",   # was fx-ex — hitting objects
    "Crushing": "fx-o",         # was fx-ex — crushing objects
    "Hammer": "fx-o",           # was fx-m — hammer is a tool, not a machine
    "Tools": "fx-o",            # was fx-m — BST fx-o includes "tools"

    # ===== ANIMALS → fx-a =====
    "Bark": "fx-a", "Cat": "fx-a", "Dog": "fx-a", "Meow": "fx-a",
    "Bird": "fx-a", "Bird_vocalization_and_bird_call_and_bird_song": "fx-a",
    "Chirp_and_tweet": "fx-a", "Crow": "fx-a", "Gull_and_seagull": "fx-a",
    "Chicken_and_rooster": "fx-a", "Fowl": "fx-a", "Frog": "fx-a",
    "Cricket": "fx-a", "Insect": "fx-a", "Growling": "fx-a",
    "Purr": "fx-a", "Wild_animals": "fx-a",
    "Domestic_animals_and_pets": "fx-a",
    "Livestock_and_farm_animals_and_working_animals": "fx-a",

    # ===== HUMAN BODY → fx-h (excluding speech) =====
    "Breathing": "fx-h", "Cough": "fx-h", "Sneeze": "fx-h",
    "Laughter": "fx-h", "Crying_and_sobbing": "fx-h",
    "Clapping": "fx-h", "Finger_snapping": "fx-h",
    "Chewing_and_mastication": "fx-h", "Burping_and_eructation": "fx-h",
    "Gasp": "fx-h", "Sigh": "fx-h", "Chuckle_and_chortle": "fx-h",
    "Giggle": "fx-h", "Walk_and_footsteps": "fx-h", "Run": "fx-h",
    "Hands": "fx-h", "Fart": "fx-h", "Yell": "fx-h", "Shout": "fx-h",
    "Screaming": "fx-h", "Respiratory_sounds": "fx-h",

    # ===== MACHINES → fx-m (machines EXCEPT vehicles) =====
    "Engine": "fx-m", "Engine_starting": "fx-m",
    "Drill": "fx-m", "Power_tool": "fx-m", "Sawing": "fx-m",
    "Mechanical_fan": "fx-m", "Clock": "fx-m",
    "Tick": "fx-m", "Tick-tock": "fx-m",
    "Mechanisms": "fx-m", "Typewriter": "fx-m", "Printer": "fx-m",
    "Ratchet_and_pawl": "fx-m",

    # ===== VEHICLES → fx-v (isolated vehicle events) =====
    # BST fx-v = "car passing by, car screeching, wiper, car brake, bike, plane, ship"
    "Accelerating_and_revving_and_vroom": "fx-v",  # was fx-m — clearly vehicle
    "Idling": "fx-v",           # was fx-m — vehicle idling
    "Car": "fx-v",              # was ss-u — single car = vehicle event
    "Car_passing_by": "fx-v",   # was ss-u — single car passing
    "Bus": "fx-v",              # was ss-u — vehicle
    "Truck": "fx-v",            # was ss-u — vehicle
    "Motorcycle": "fx-v",       # was ss-u — vehicle
    "Motor_vehicle_(road)": "fx-v",  # was ss-u — vehicle
    "Vehicle_horn_and_car_horn_and_honking": "fx-v",  # was ss-u — vehicle sound
    "Train": "ss-u",            # train passing is usually ambient soundscape
    "Rail_transport": "ss-u",   # rail = ambient soundscape
    "Boat_and_Water_vehicle": "fx-v",  # water vehicle = vehicle event
    "Fixed-wing_aircraft_and_airplane": "fx-v",  # was ss-u — vehicle
    "Aircraft": "fx-v",         # was ss-u — vehicle
    "Skateboard": "fx-v",       # was ss-u — transport device
    "Bicycle": "fx-v",          # was ss-u — vehicle

    # ===== NATURAL ELEMENTS → fx-n =====
    "Drip": "fx-n", "Pour": "fx-n", "Fill_(with_liquid)": "fx-n",
    "Splash_and_splatter": "fx-n", "Trickle_and_dribble": "fx-n",
    "Water_tap_and_faucet": "fx-n", "Sink_(filling_or_washing)": "fx-n",
    "Fire": "fx-n", "Crackle": "fx-n", "Hiss": "fx-n",
    "Boiling": "fx-n", "Gurgling": "fx-n",

    # ===== EVERYDAY OBJECTS → fx-o =====
    "Door": "fx-o", "Cupboard_open_or_close": "fx-o",
    "Drawer_open_or_close": "fx-o", "Sliding_door": "fx-o",
    "Knock": "fx-o", "Coin_(dropping)": "fx-o",
    "Computer_keyboard": "fx-o", "Typing": "fx-o",
    "Keys_jangling": "fx-o", "Dishes_and_pots_and_pans": "fx-o",
    "Cutlery_and_silverware": "fx-o", "Scissors": "fx-o",
    "Crumpling_and_crinkling": "fx-o", "Tearing": "fx-o",
    "Packing_tape_and_duct_tape": "fx-o", "Glass": "fx-o",
    "Zipper_(clothing)": "fx-o", "Camera": "fx-o",
    "Toilet_flush": "fx-o", "Writing": "fx-o", "Wood": "fx-o",
    "Squeak": "fx-o", "Rattle": "fx-o", "Tap": "fx-o",
    "Bathtub_(filling_or_washing)": "fx-o", "Frying_(food)": "fx-o",
    "Scratching_(performance_technique)": "fx-o",
    "Chink_and_clink": "fx-o",

    # ===== ELECTRONIC/DESIGNED → fx-el =====
    # BST fx-el = "sci-fi, laser, whoosh, boink, cartoon, UI, sound alerts or notifications"
    "Alarm": "fx-el", "Ringtone": "fx-el", "Doorbell": "fx-el",
    "Telephone": "fx-el", "Buzz": "fx-el",
    "Whoosh_and_swoosh_and_swish": "fx-el",  # was fx-o — BST lists "whoosh" under fx-el

    # ===== NATURE SOUNDSCAPES → ss-n =====
    "Rain": "ss-n", "Raindrop": "ss-n", "Thunder": "ss-n",
    "Thunderstorm": "ss-n", "Wind": "ss-n", "Ocean": "ss-n",
    "Waves_and_surf": "ss-n", "Stream": "ss-n",

    # ===== URBAN SOUNDSCAPES → ss-u (continuous ambient, NOT single events) =====
    "Traffic_noise_and_roadway_noise": "ss-u",  # continuous traffic = soundscape
    "Siren": "ss-u",           # siren in urban context = urban soundscape

    # ===== SPEECH =====
    "Male_speech_and_man_speaking": "sp-s",
    "Female_speech_and_woman_speaking": "sp-s",
    "Child_speech_and_kid_speaking": "sp-s",
    "Speech_synthesizer": "sp-p",

    # ===== INSTRUMENT SAMPLES (single notes, NOT music excerpts) =====
    "Drum": "is-p", "Drum_kit": "is-p", "Bass_drum": "is-p",
    "Snare_drum": "is-p", "Hi-hat": "is-p", "Crash_cymbal": "is-p",
    "Cymbal": "is-p", "Tambourine": "is-p", "Cowbell": "is-p",
    "Tabla": "is-p", "Gong": "is-p", "Mallet_percussion": "is-p",
    "Marimba_and_xylophone": "is-p", "Glockenspiel": "is-p",
    "Rattle_(instrument)": "is-p", "Percussion": "is-p",
    "Acoustic_guitar": "is-s", "Guitar": "is-s", "Strum": "is-s",
    "Bass_guitar": "is-s", "Bowed_string_instrument": "is-s",
    "Plucked_string_instrument": "is-s", "Harp": "is-s",
    "Electric_guitar": "is-s",
    "Trumpet": "is-w", "Brass_instrument": "is-w",
    "Harmonica": "is-w", "Accordion": "is-w",
    "Wind_instrument_and_woodwind_instrument": "is-w",
    "Piano": "is-k", "Keyboard_(musical)": "is-k", "Organ": "is-k",
}

# Synthetic text descriptions for CLAP text embedding extraction
# These must match BST definitions exactly
BST_SYNTHETIC_TEXT = {
    "fx-a": "animal sound recording",
    "fx-h": "human body sound effect breath sneeze clapping walking",
    "fx-m": "mechanical machine sound effect drill gear motor",
    "fx-n": "natural element sound effect water fire explosion ice crack stone splash",
    "fx-o": "everyday object sound effect tool weapon impact door",
    "fx-el": "electronic designed sound effect alert notification whoosh sci-fi UI",
    "fx-ex": "experimental reversed weird unusual processed audio effect",
    "fx-v": "vehicle sound effect car passing plane engine bike ship",
    "ss-n": "nature outdoor environment soundscape field recording forest river",
    "ss-u": "urban city street traffic soundscape ambience busy road",
    "ss-i": "indoor room interior ambience soundscape quiet office",
    "ss-s": "synthetic artificial digital soundscape imaginary",
    "sp-s": "single person speaking voice speech",
    "sp-c": "crowd people talking conversation multiple voices dialogue",
    "sp-p": "processed speech podcast radio broadcast phone recording",
    "m-sp": "solo percussion music drum passage rhythm loop",
    "m-si": "single instrument music solo performance melody",
    "m-m": "multiple instruments music ensemble band orchestra",
    "is-p": "percussion drum hit cymbal bell single note sample",
    "is-s": "string instrument single note guitar violin pluck bow",
    "is-w": "wind instrument single note flute trumpet brass reed",
    "is-k": "keyboard piano organ single note sample",
    "is-e": "electronic synthesizer single note sample digital",
}
