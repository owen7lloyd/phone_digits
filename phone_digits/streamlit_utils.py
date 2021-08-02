"""
Utils for streamlit app that needs session_state access
"""

import streamlit as st
import numpy as np
import random
import base64

from pickle import dumps
from taped import LiveWf
from utils import (
    get_cont_intervals,
    plot_wf,
    barplot_colored,
    mk_event_location_plot,
    barplot_thresh,
)


# -------------------------------DOWNLOAD UTILS-------------------------------


def download_model(model, model_type, col):
    """
    Hack to download model with streamlit shairng
    """
    output_model = dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="{model_type}.pkl">Download trained {model_type}.pkl file</a>'
    col.markdown(href, unsafe_allow_html=True)


# -------------------------------STREAMLIT UTILS-------------------------------


def store_in_ss(key, value):
    """
    Stores the key-value pair passed in st.session_state if the key is not already present
    """
    if key not in st.session_state:
        st.session_state[key] = value


def annotations():
    """
    Helper widget to generate annotations to optimize the threshold
    """
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
    # dir = st.session_state.zip_path[
    #     : -len(st.session_state.zip_path.split("/")[-1]) - 1
    # ]
    # save_audio_for_st(
    #     np.hstack(
    #         st.session_state.chks[
    #             st.session_state.current_chk - 1 : st.session_state.current_chk + 1
    #         ]
    #     ),
    #     dir,
    # )
    # col1.audio(f"{dir}/chk.wav")

    with col1:
        st.pyplot(
            plot_wf(st.session_state.chks[st.session_state.current_chk], linewidth=0.8)
        )
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
    """
    Writes the number of continuous intervals at a given thresh value
    """
    st.write(
        f"There are {len(get_cont_intervals(thresh_df[thresh_df['meets_thresh'] == 1].index))} "
        f"continuous intervals with thresh at {thresh}."
    )


def clear_recording():
    """
    Clears the current recording from st.session_state and TestController
    """
    st.session_state.TestController.clear_recording()

    keys = ["wf", "results", "test_fvs", "number", "thresh_adjust"]
    for key in keys:
        if key in st.session_state:
            st.session_state.__delattr__(key)


# -------------------------------TAPED UTILS-------------------------------


def record(input_device, length):
    """
    Creates a waveform from a recording generated with given input device and length
    """
    with LiveWf(input_device) as live_audio_stream:
        wf = np.array(live_audio_stream[22_050 : 22_050 + length * 44_100])
    st.session_state.wf = wf


def stop():
    """
    Stops the recording for livestream data
    """
    st.session_state.pressed = False


# -------------------------------MODEL UTILS-------------------------------


def show_adjustment(container):
    """
    Show the affect of an adjustment to threshold level on continous intervals
    """
    with container:
        thresh_df, barplot = st.session_state.TestController.adjust_thresh(
            st.session_state.thresh_adjust
        )
        write_intervals(
            thresh_df,
            st.session_state.TestController.threshold + st.session_state.thresh_adjust,
        )
        st.pyplot(barplot)


def run_model():
    """
    Run the model on the test waveform
    """
    test_fvs, results, phone_number = st.session_state.TestController.test_model(
        st.session_state.thresh_adjust
    )
    st.session_state.thresh_fvs = test_fvs
    st.session_state.results = results
    st.session_state.number = phone_number


def display_results(final=True):
    """
    Display the results using a colored barplot and event location plot
    """
    if final:
        st.pyplot(
            barplot_colored(st.session_state.thresh_fvs, st.session_state.results)
        )
        st.pyplot(mk_event_location_plot(st.session_state.results))
        st.success(
            "The detected phone digits are "
            + "".join([str(num) for num in st.session_state.number])
        )
    else:
        st.pyplot(st.session_state.plt, clear_figure=False)
        st.pyplot(mk_event_location_plot(st.session_state.results))
        st.write("".join([str(num) for num in st.session_state.number]))
