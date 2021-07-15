import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from scipy.optimize import minimize_scalar
from pickle import dump, load

from lined import LineParametrized
from slang import fixed_step_chunker
from recode import ChunkedEncoder, ChunkedDecoder, StructCodecSpecs
from py2store import LocalBinaryStore


# -------------------------------DEFAULTS-------------------------------


DFLT_CHK_SIZE = 2048
DFLT_CHK_STEP = 2048
DFLT_SR = 44100

FEATURE_MAP = {
    "Principle Component Analysis (PCA)": PCA,
    "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis,
    "T-Distributed Stochastic Neighbour Embedding (T-SNE)": TSNE,
}

MODEL_MAP = {
    "Random Forest": RandomForestClassifier,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Support Vector Machine": SVC,
}


# -------------------------------RECODE + DOL-------------------------------


def encode_dol(featurizer, model, fvs, tags, thresh, file_dir):
    n_channels = fvs.shape[1]
    fvs_specs = StructCodecSpecs(chk_format="d", n_channels=n_channels)
    tags_specs = StructCodecSpecs(chk_format="h")
    thresh_specs = StructCodecSpecs(chk_format="d")

    fvs_encoder = ChunkedEncoder(fvs_specs.frame_to_chk)
    tags_encoder = ChunkedEncoder(tags_specs.frame_to_chk)
    thresh_encoder = ChunkedEncoder(thresh_specs.frame_to_chk)

    fvs_b = fvs_encoder(fvs)
    tags_b = tags_encoder(tags)
    n_channels_b = tags_encoder([n_channels])
    thresh_b = thresh_encoder([thresh])

    bstore = LocalBinaryStore(file_dir)
    bstore["fvs"] = fvs_b
    bstore["tags"] = tags_b
    bstore["n_channels"] = n_channels_b
    bstore["thresh"] = thresh_b

    with open(f"{file_dir}/featurizer.pkl", "wb") as f:
        dump(featurizer, f)

    with open(f"{file_dir}/model.pkl", "wb") as m:
        dump(model, m)


def decode_dol(file_dir):
    tags_specs = StructCodecSpecs(chk_format="h")
    thresh_specs = StructCodecSpecs(chk_format="d")
    tags_decoder = ChunkedDecoder(tags_specs.chk_to_frame)
    thresh_decoder = ChunkedDecoder(thresh_specs.chk_to_frame)

    bstore = LocalBinaryStore(file_dir)
    decoded_tags = np.array(tags_decoder(bstore["tags"]))
    decoded_n_channels = np.array(tags_decoder(bstore["n_channels"]))[0]
    decoded_thresh = np.array(thresh_decoder(bstore["thresh"]))[0]

    fvs_specs = StructCodecSpecs(chk_format="d", n_channels=decoded_n_channels)
    fvs_decoder = ChunkedDecoder(fvs_specs.chk_to_frame)
    decoded_fvs = np.array(fvs_decoder(bstore["fvs"]))

    with open(f"{file_dir}/featurizer.pkl", "rb") as f:
        featurizer = load(f)

    with open(f"{file_dir}/model.pkl", "rb") as m:
        model = load(m)

    return decoded_fvs, decoded_tags, decoded_thresh, featurizer, model


# -------------------------------ANNOTATION UTILS-------------------------------


def save_audio_for_st(data, dir, sr=DFLT_SR):
    sf.write(f"{dir}/chk.wav", data, sr)
    return f"{dir}/chk.wav"


# -------------------------------TRAINING UTILS-------------------------------


def chk_tag_gen(wf_tag_iter, chunker):
    for wf, tag in wf_tag_iter:
        for chk in chunker(wf):
            yield chk, tag


def mk_thresh_df(wfs_and_tags, thresh):
    return pd.DataFrame(
        intensity_gen(
            wfs_and_tags=wfs_and_tags,
            thresh=thresh,
        ),
        columns=["meets_thresh", "chk"],
    )


def mk_thresh_chks(thresh_df):
    return np.array(thresh_df[thresh_df.meets_thresh == 1].chk)


def chks_to_spectra(chks):
    return list(map(lambda chk: np.abs(np.fft.rfft(chk)), chks))


def mk_chks_and_tags(dacc, chunker, thresh_chks):
    chks, tags = list(), list()
    for idx, chk_tag in enumerate(list(dacc.chk_tag_gen(chunker))):
        if idx in thresh_chks:
            chks.append(chk_tag[0])
            tags.append(chk_tag[1])
    return chks, tags


def mk_featurizer(chks, tags, featurizer_choice):
    spectra = chks_to_spectra(chks)
    featurizer = FEATURE_MAP[featurizer_choice](n_components=9)
    featurizer.fit(spectra, tags)
    fvs = featurizer.transform(spectra)
    return fvs, featurizer


def mk_model(fvs, tags, model_choice):
    model = MODEL_MAP[model_choice]()
    return model.fit(fvs, tags)


# -------------------------------TESTING UTILS-------------------------------


def get_cont_intervals(indices_list):
    intervals = []
    if len(indices_list) == 0:
        return []
    current_item = indices_list[0]
    inter = [current_item]
    for next_item in indices_list[1:]:
        if next_item == current_item + 1:
            inter.append(next_item)
        else:
            intervals.append(inter)
            inter = [next_item]
        current_item = next_item
    if len(inter) > 0:
        intervals.append(inter)
    return intervals


def normalize_wf(wf):
    new_wf = np.float32(np.array(wf)).reshape(1, -1)
    return normalize(new_wf)[0]


def mk_chks_and_noise_tags(chks_and_tags_enum, thresh_chks):
    chks = []
    noise_tags = {}
    for idx, chk_tag in chks_and_tags_enum:
        if idx in thresh_chks:
            noise_tags[idx] = False
            chks.append(chk_tag[0])
        else:
            noise_tags[idx] = True
    return chks, noise_tags


def mk_results(noise_tags, scores):
    results = []
    for idx in range(len(noise_tags)):
        if noise_tags[idx]:
            results.append(None)
        else:
            results.append(scores.pop(0))
    return results


def mk_number(results):
    i = 0
    number = []

    if results[0] is not None and results[1] is not None:
        pressed = True
        tags = [results[0], results[1]]
        j = 1
        while pressed:
            if j == len(results) - 1:
                break
            if results[j + 1] is not None:
                tags.append(results[j + 1])
            else:
                pressed = False
            j += 1
        if len(tags) > 2:
            number.append(max(set(tags), key=tags.count))

    while i < len(results) - 1:
        if results[i] is None and results[i + 1] is not None:
            pressed = True
            tags = [results[i + 1]]
            j = i + 1
            while pressed:
                if j == len(results) - 1:
                    break
                if results[j + 1] is not None:
                    tags.append(results[j + 1])
                else:
                    pressed = False
                j += 1
            if len(tags) > 2:
                number.append(max(set(tags), key=tags.count))
        i += 1
    return number


def charts_for_live(live_wf, chk_stop, chunker, temp_gen, thresh, featurizer, model):
    chks = []
    chks_and_tags = []
    new_wf = normalize_wf(live_wf[0:chk_stop])

    for chk in chunker(new_wf):
        chks.append(chk)
        chks_and_tags.append([chk, 11])

    test_thresh_df = pd.DataFrame(
        temp_gen(
            chks=chks,
            thresh=thresh,
        ),
        columns=["meets_thresh", "chk"],
    )
    test_thresh_chks = mk_thresh_chks(test_thresh_df)

    new_chks, noise_tags = mk_chks_and_noise_tags(
        enumerate(chks_and_tags), test_thresh_chks
    )

    test_fvs = featurizer.transform(chks_to_spectra(new_chks))
    thresh_fvs = list(threshold_featurizer(chks))

    scores = list(model.predict(test_fvs))
    results = mk_results(noise_tags, scores)
    number = mk_number(results)
    bar_plt = barplot_colored(thresh_fvs, results)

    return bar_plt, results, number, thresh_fvs


# -------------------------------PLOTTING UTILS-------------------------------


def plot_wf(wf):
    plt.figure()
    plt.plot(wf, linewidth=0.2)
    plt.ylim(bottom=-0.0075, top=0.0075)
    return plt


def barplot_thresh(
    fvs,
    thresh,
):
    plt.figure()
    plt.bar(list(range(len(fvs))), fvs)
    plt.ylim(bottom=0, top=0.0075)
    plt.plot([thresh] * len(fvs), color="r")
    return plt


def barplot_colored(fvs, results):
    plt.figure()
    times = np.array(range(len(list(fvs)))) * (2048 / 44100)

    for result in set(results):
        plt.bar(
            np.take(times, np.where(results == result)[0]),
            np.take(fvs, np.where(results == result)[0]),
            label=result,
            width=0.03,
        )

    plt.legend()
    plt.ylim(bottom=0, top=0.0075)
    return plt


def mk_event_location_plot(tags):
    new_tags = []
    for tag in tags:
        new_tags.append(str(tag))
    x = np.array(list(range(len(new_tags))))
    y = np.array(new_tags)
    length = len(tags) * 2048 / 44100
    times = np.arange(0, 0.5 + round(length * 2) / 2, 0.5)
    ticks = np.around(times * 44100 / 2048)

    labels = set(tags)
    plt.figure(figsize=(12, 4))
    plt.barh(y, [1] * len(x), left=x, color="purple", align="center", height=1)
    plt.xticks(ticks, times)
    plt.yticks(np.arange(len(labels)), labels)
    return plt


# -------------------------------SMART CHUNKING-------------------------------


def threshold_chunker(wfs_and_tags, chk_size=2048, chk_step=2048):
    for wf, tag in wfs_and_tags:
        for chk in fixed_step_chunker(wf, chk_size=chk_size, chk_step=chk_step):
            yield chk


def threshold_featurizer(chks):
    for chk in chks:
        yield np.max(np.absolute(chk))


def threshold_model(fvs, thresh):
    for idx, fv in enumerate(fvs):
        if fv > thresh:
            yield 1, idx
        else:
            yield 0, idx


intensity_gen = LineParametrized(
    threshold_chunker, threshold_featurizer, threshold_model
)


class MetricOptimizedThreshold(BaseEstimator):
    def __init__(self, bounds=[0, 0.006], metric=accuracy_score):
        self.bounds = bounds
        self.metric = metric

    def fit(self, X, y):
        chk_ids = sorted(list(y.keys()))
        truth = [y[key] for key in chk_ids]

        def opt_func(thresh):
            thresh_results = np.array(
                list(intensity_gen(wfs_and_tags=X, thresh=thresh))
            )
            pred = [int(thresh_results[chk_id][0]) for chk_id in chk_ids]
            score = self.metric(truth, pred)
            return -1.0 * score

        optimize_result = minimize_scalar(
            fun=opt_func,
            method="bounded",
            bounds=self.bounds,
        )

        self.thresh_ = optimize_result["x"]

        return self

    def score(self):
        return self.thresh_
