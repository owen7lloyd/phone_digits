"""
Controller for streamlit app that needs session_state access
"""

import streamlit as st
import numpy as np
import random

from functools import partial

from slang import fixed_step_chunker
from taped import LiveWf

from controller import (
    get_cont_intervals,
    DFLT_CHK_SIZE,
    DFLT_CHK_STEP,
    chk_tag_gen,
    save_audio_for_st,
    mk_thresh_df,
    barplot_thresh,
    mk_thresh_chks,
    chks_to_spectra,
    mk_chks_and_noise_tags,
    mk_results,
    mk_number,
    mk_chks_and_tags,
    mk_featurizer,
    mk_model,
)

# -------------------------------STREAMLIT UTILS-------------------------------


def store_in_ss(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def annotations():
    if "annotations" not in st.session_state:
        st.session_state.annotations = {}
        st.session_state.current_chk = st.session_state.random_chk_ids[0]

    def annotate(label):
        st.session_state.annotations[int(st.session_state.current_chk)] = label
        if len(st.session_state.random_chk_ids) > 0:
            st.session_state.current_chk = random.choice(
                st.session_state.random_chk_ids
            )
            st.session_state.random_chk_ids = np.delete(
                st.session_state.random_chk_ids,
                np.where(
                    st.session_state.random_chk_ids == st.session_state.current_chk
                ),
            )

    st.write("")
    col1, col2 = st.beta_columns(2)
    dir = st.session_state.zip_path[
        : -len(st.session_state.zip_path.split("/")[-1]) - 1
    ]
    save_audio_for_st(
        np.hstack(
            st.session_state.chks[
                st.session_state.current_chk - 1 : st.session_state.current_chk + 1
            ]
        ),
        dir,
    )
    col1.audio(f"{dir}/chk.wav")
    with col2:
        if len(st.session_state.random_chk_ids) > 0:
            st.write(
                "Annotated:",
                len(st.session_state.annotations),
                "- Remaining:",
                len(st.session_state.random_chk_ids),
            )
            st.button("This is a digit click!", on_click=annotate, args=(1,))
            st.button("This is background noise!", on_click=annotate, args=(0,))
        else:
            st.success(f"🎈 Done! All the sample chunks are annotated.")
            return True


def write_intervals(thresh_df, thresh):
    st.write(
        f"There are {len(get_cont_intervals(thresh_df[thresh_df['meets_thresh'] == 1].index))} "
        f"continuous intervals with thresh at {thresh}."
    )


def clear_recording():
    keys = ["wf", "test_wf_tag_gen", "thresh_adjust", "numbers"]
    for key in keys:
        if key in st.session_state:
            st.session_state.__delattr__(key)


def mk_featurizer_and_model(featurizer_choice, model_choice):
    chks, tags = mk_chks_and_tags(
        st.session_state.dacc,
        partial(
            fixed_step_chunker,
            chk_size=DFLT_CHK_SIZE,
            chk_step=DFLT_CHK_STEP,
        ),
        st.session_state.thresh_chks,
    )
    fvs, featurizer = mk_featurizer(chks, tags, featurizer_choice)
    model = mk_model(fvs, tags, model_choice)

    st.session_state.featurizer = featurizer
    st.session_state.model = model
    st.session_state.fvs = fvs
    st.session_state.tags = tags


# -------------------------------TAPED UTILS-------------------------------


def record(input_device, length):
    with LiveWf(input_device) as live_audio_stream:
        wf = np.array(live_audio_stream[22_050 : 22_050 + length * 44_100])
    st.session_state.wf = wf


def stop():
    st.session_state.pressed = False


# -------------------------------MODEL UTILS-------------------------------


def adjust_thresh(container):
    with container:
        test_thresh_df = mk_thresh_df(
            st.session_state.test_wf_tag_gen,
            st.session_state.thresh + st.session_state.thresh_adjust,
        )
        write_intervals(
            test_thresh_df, st.session_state.thresh + st.session_state.thresh_adjust
        )
        st.pyplot(
            barplot_thresh(
                st.session_state.test_fvs,
                st.session_state.thresh + st.session_state.thresh_adjust,
            )
        )


def run_model():
    test_thresh_df = mk_thresh_df(
        st.session_state.test_wf_tag_gen,
        st.session_state.thresh + st.session_state.thresh_adjust,
    )
    test_thresh_chks = mk_thresh_chks(test_thresh_df)

    chunker = partial(
        fixed_step_chunker, chk_size=DFLT_CHK_SIZE, chk_step=DFLT_CHK_STEP
    )

    chks_and_tags_enum = list(
        enumerate(list(chk_tag_gen(st.session_state.test_wf_tag_gen, chunker)))
    )

    chks, noise_tags = mk_chks_and_noise_tags(chks_and_tags_enum, test_thresh_chks)

    spectra = chks_to_spectra(chks)
    test_fvs = st.session_state.featurizer.transform(spectra)
    scores = list(st.session_state.model.predict(test_fvs))

    st.session_state.results = mk_results(noise_tags, scores)
    st.session_state.numbers = mk_number(st.session_state.results)
