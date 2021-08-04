"""
Controller for streamlit app
"""

import numpy as np
import pandas as pd

from phone_digits_dacc import mk_dacc
from utils import (
    threshold_chunker,
    threshold_featurizer,
    mk_thresh_df,
    mk_thresh_chks,
    MetricOptimizedThreshold,
    DFLT_CHUNKER,
    chks_to_spectra,
    FEATURE_MAP,
    MODEL_MAP,
    normalize_wf_taped,
    normalize_wf_upload,
    plot_wf,
    chk_tag_gen,
    mk_chks_and_noise_tags,
    mk_results,
    mk_number,
    barplot_colored,
    threshold_model,
    barplot_thresh,
)


class SmartChunkingController:
    def __init__(self, zip_dir):
        self.zip_dir = zip_dir

    def mk_dacc(self):
        pass

    def mk_featurizer(self):
        pass

    def mk_model(self):
        pass

    def train_thresh_model(self):
        pass

    def score(self):
        pass


class PhoneDigitsController(SmartChunkingController):
    def __init__(self, zip_dir):
        super().__init__(zip_dir)
        self.dacc = mk_dacc(self.zip_dir)

    def mk_annotations_data(self):
        """
        Generate the data needed to create annotations to optimize the threshold for smart chunking
        """
        wf_tag_it = self.dacc.wf_tag_gen()
        thresh_chks = list(threshold_chunker(wf_tag_it))
        thresh_chk_ids = np.array(
            list(enumerate(np.ones(len(thresh_chks)))), dtype=int
        )[:, 0]
        random_thresh_chk_ids = np.random.choice(thresh_chk_ids, 5)
        return thresh_chks, thresh_chk_ids, random_thresh_chk_ids

    def train_thresh_model(self, annots):
        """
        Optimize the threshold for smart chunking with the annotations data
        """
        self.mot = MetricOptimizedThreshold().fit(list(self.dacc.wf_tag_gen()), annots)

    def score(self):
        """
        Get the optimized threshold for smart chunking
        """
        return self.mot.score()

    def mk_train_thresh_df(self):
        """
        Returns a dataframe indicating which chunks in the training set exceed the optimized threshold
        """
        self.thresh_df = mk_thresh_df(self.dacc.wf_tag_gen(), self.mot.score())

    def mk_train_thresh_chks(self):
        """
        Returns the chunks in the training set that exceed the optimized threshold
        """
        self.thresh_chks = mk_thresh_chks(self.thresh_df)

    def mk_chks_and_tags(self):
        """
        Generates chunks and tags for each waveform in the training set
        """
        chks, tags = list(), list()
        chunker = DFLT_CHUNKER
        for idx, chk_tag in enumerate(list(self.dacc.chk_tag_gen(chunker))):
            if idx in self.thresh_chks:
                chks.append(chk_tag[0])
                tags.append(chk_tag[1])
        self.chks = chks
        self.tags = tags

    def mk_featurizer(self, featurizer_choice):
        """
        Trains a featurizer on the training set and generates fvs based on the featurizer_choice determined by the user
        """
        self.mk_chks_and_tags()
        spectra = chks_to_spectra(self.chks)
        featurizer = FEATURE_MAP[featurizer_choice](
            n_components=len(set(self.tags)) - 1
        )
        self.featurizer = featurizer.fit(spectra, self.tags)
        self.fvs = self.featurizer.transform(spectra)

    def mk_model(self, model_choice):
        """
        Trains a model on the training set and generates predictions based on the model_choice determined by the user
        """
        model = MODEL_MAP[model_choice]()
        self.model = model.fit(self.fvs, self.tags)
        self.scores = self.model.predict(self.fvs)

    def params_for_test(self):
        """
        Returns the trained featurizer, model, and optimized threshold value to initialize a PhoneDigitsTestController
        """
        return self.featurizer, self.model, self.score()


class PhoneDigitsTestController:
    def __init__(self, featurizer, model, threshold):
        self.featurizer = featurizer
        self.model = model
        self.threshold = threshold

    def __delattr__(self, item):
        del self.__dict__[item]

    def set_wf(self, wf, upload=True):
        """
        Normalize the waveform to get results for
        """
        if upload:
            self.wf = normalize_wf_upload(wf)
        else:
            self.wf = normalize_wf_taped(wf)

    def plot_wf(self):
        """
        Plot the waveform we are interested in getting results for
        """
        return plot_wf(self.wf)

    def prep_for_test(self):
        """
        Prepare the fvs for the chunks in the test waveform that exceed the optimized threshold
        """
        self.wf_tag_gen = [(self.wf, 11)]
        test_chks = list(threshold_chunker(self.wf_tag_gen))
        self.test_fvs = list(threshold_featurizer(test_chks))
        self.test_thresh_df = mk_thresh_df(self.wf_tag_gen, self.threshold)

    def barplot_thresh(self, adjustment=0):
        """
        Returns a matplotlib.pyplot fig of a barplot of the fvs for each chunk in the test waveform with a display
        of the optimal threshold level
        """
        return barplot_thresh(self.test_fvs, self.threshold + adjustment)

    def adjust_thresh(self, adjustment):
        """
        Returns a matplotlib.pyplot fig of a barplot of the fvs for each chunk in the test waveform with a display
        of the optimal threshold level adjusted by the passed argument
        """
        adjusted_thresh = self.threshold + adjustment
        return mk_thresh_df(self.wf_tag_gen, adjusted_thresh), self.barplot_thresh(
            adjustment
        )

    def test_model(self, adjustment):
        """
        Generate a prediction for the test waveform as well as other values for creating visuals
        """
        adjusted_thresh = self.threshold + adjustment
        test_thresh_df = mk_thresh_df(self.wf_tag_gen, adjusted_thresh)
        test_thresh_chks = mk_thresh_chks(test_thresh_df)
        chks_and_tags_enum = list(
            enumerate(list(chk_tag_gen(self.wf_tag_gen, DFLT_CHUNKER)))
        )
        test_chks, noise_tags = mk_chks_and_noise_tags(
            chks_and_tags_enum, test_thresh_chks
        )
        spectra = chks_to_spectra(test_chks)
        test_fvs = self.featurizer.transform(spectra)
        scores = list(self.model.predict(test_fvs))
        results = mk_results(noise_tags, scores)
        phone_number = mk_number(results)
        return self.test_fvs, results, phone_number

    def charts_for_live(self, live_wf, chk_stop):
        """
        Generate displays for a LiveWf from taped
        """
        chks = []
        chks_and_tags = []
        new_wf = normalize_wf_taped(live_wf[0:chk_stop])

        for chk in DFLT_CHUNKER(new_wf):
            chks.append(chk)
            chks_and_tags.append([chk, 11])

        test_thresh_df = pd.DataFrame(
            threshold_model(
                threshold_featurizer(chks),
                self.threshold,
            ),
            columns=["meets_thresh", "chk"],
        )
        test_thresh_chks = mk_thresh_chks(test_thresh_df)

        new_chks, noise_tags = mk_chks_and_noise_tags(
            enumerate(chks_and_tags), test_thresh_chks
        )

        spectra = chks_to_spectra(chks)
        test_fvs = self.featurizer.transform(spectra)
        thresh_fvs = list(threshold_featurizer(chks))

        scores = list(self.model.predict(test_fvs))
        results = mk_results(noise_tags, scores)
        number = mk_number(results)
        bar_plt = barplot_colored(thresh_fvs, results)

        return bar_plt, results, number, thresh_fvs

    def clear_recording(self):
        """
        Reset the controller for a new recording
        """
        keys = ["wf", "wf_tag_gen", "test_fvs", "test_thresh_df"]
        for key in keys:
            self.__delattr__(key)
