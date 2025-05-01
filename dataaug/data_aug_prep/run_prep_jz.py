from musan_prepare import prepare_musan


base_save="/lustre/fsn1/projects/rech/nkp/uaj64gk/african_brq/MUSAN"

musan_folder = "/lustre/fsn1/projects/rech/nkp/uaj64gk/african_brq/MUSAN"
music_csv = base_save + "/music.csv"
noise_csv = base_save + "/noise.csv"
speech_csv = base_save + "/speech.csv"
example_length = None

prepare_musan(
    folder=musan_folder,
    music_csv=music_csv,
    noise_csv=noise_csv,
    speech_csv=speech_csv,
    max_noise_len=example_length
)