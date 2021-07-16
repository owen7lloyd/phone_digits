"""
Streamlit app for phone_digit detection
"""

import streamlit as st
import numpy as np

from functools import partial

from slang import fixed_step_chunker
from taped import list_recording_device_index_names, LiveWf

from phone_digits_dacc import mk_dacc
from streamlit_controller import (
    store_in_ss,
    annotations,
    adjust_thresh,
    clear_recording,
    run_model,
    record,
    write_intervals,
    stop,
    mk_featurizer_and_model,
    download_model,
)
from controller import (
    threshold_chunker,
    threshold_featurizer,
    encode_dol,
    decode_dol,
    DFLT_CHK_SIZE,
    DFLT_CHK_STEP,
    MetricOptimizedThreshold,
    mk_thresh_df,
    mk_thresh_chks,
    normalize_wf,
    plot_wf,
    barplot_thresh,
    barplot_colored,
    charts_for_live,
    mk_event_location_plot,
)

st.title("Phone Digits")

train = st.selectbox("Would you like to train a new model?", ["Yes", "No"])

if train == "Yes":
    st.session_state.zip_file = st.file_uploader(
        label="Please upload the phone_digits training data!"
    )
    # st.session_state.zip_path = st.text_input(
    #     "What is your path to the phone_digits training data?",
    #     "" if "zip_path" not in st.session_state else st.session_state.zip_path,
    # )

    st.write(
        "If you do not have the phone_digits training "
        "data it can be found [here](https://www.dropbox.com/s/irbaaopwy1cpv5x/phone_digits_train.zip?dl=0)."
    )

    if st.session_state.zip_file:
        store_in_ss("dacc", mk_dacc(st.session_state.zip_file))
        store_in_ss("wfs_and_tags", list(st.session_state.dacc.wf_tag_gen()))
        store_in_ss("chks", list(threshold_chunker(st.session_state.wfs_and_tags)))
        store_in_ss("thresh_fvs", list(threshold_featurizer(st.session_state.chks)))
        store_in_ss(
            "chk_ids",
            np.array(list(enumerate(np.ones(len(st.session_state.chks)))), dtype=int)[
                :, 0
            ],
        )
        store_in_ss("random_chk_ids", np.random.choice(st.session_state.chk_ids, 1))
        st.markdown("""---""")
        st.markdown("### Annotations")
        st.session_state.annotated = annotations()

        if st.session_state.annotated:
            st.markdown("""---""")

            col1, col2 = st.beta_columns(2)
            with col1:
                featurizer_choice = st.selectbox(
                    "Which featurizer would you like to use?",
                    options=[
                        "Principle Component Analysis (PCA)",
                        "Linear Discriminant Analysis (LDA)",
                        "T-Distributed Stochastic Neighbour Embedding (T-SNE)",
                    ],
                )
            with col2:
                model_choice = st.selectbox(
                    "Which model would you like to use?",
                    options=[
                        "Support Vector Machine",
                        "Random Forest",
                        "K-Nearest Neighbors",
                    ],
                )
            store_in_ss(
                "thresh",
                MetricOptimizedThreshold()
                .fit(st.session_state.wfs_and_tags, st.session_state.annotations)
                .score(),
            )
            store_in_ss(
                "thresh_df",
                mk_thresh_df(
                    st.session_state.wfs_and_tags,
                    st.session_state.thresh,
                ),
            )
            store_in_ss("thresh_chks", mk_thresh_chks(st.session_state.thresh_df))

            st.markdown("""---""")

            st.button(
                "Click here to build your featurizer and model",
                on_click=mk_featurizer_and_model,
                args=(featurizer_choice, model_choice),
            )

        if "model" in st.session_state and "featurizer" in st.session_state:
            st.button(
                "Click here to save your model",
                on_click=download_model,
                args=(st.session_state.model, 'model'),
            )
            st.button(
                "Click here to save your featurizer",
                on_click=download_model,
                args=(st.session_state.featurizer, 'featurizer'),
            )


if train == "No":
    st.session_state.dir = st.text_input(
        "Where are your persisted fvs and tags?",
        "" if "dir" not in st.session_state else st.session_state.dir,
    )

    if st.session_state.dir != "":
        fvs, tags, thresh, featurizer, model = decode_dol(st.session_state.dir)
        store_in_ss("fvs", fvs)
        store_in_ss("tags", tags)
        store_in_ss("thresh", thresh)
        store_in_ss("featurizer", featurizer)
        store_in_ss("model", model)
        st.success(f"🎈 Done! Your featurizer and model have been successfully decoded!")

if "fvs" in st.session_state:
    st.markdown("""---""")

    scores = st.session_state.model.predict(st.session_state.fvs)
    st.success(
        f"The model was able to correctly classify "
        f"{100 * np.sum(scores == st.session_state.tags) / len(st.session_state.tags)}% "
        f"of the digits pressed in the training data"
    )

if "model" in st.session_state:
    st.markdown("""---""")

    st.write(
        "If you are planning to use the live output option, "
        "it is recommended that you restart the page and select to not train a new model for better performance."
    )

    st.session_state.live = st.selectbox(
        "Would you prefer to make a recording or see results live?",
        ["Recording", "Live"],
    )


if "live" in st.session_state:
    mic = st.selectbox(
        "Which microphone would you like to use?",
        list_recording_device_index_names(),
    )
    if st.session_state.live == "Recording":
        time = st.number_input(
            "How long would you like the recording to be?",
            min_value=0,
            max_value=15,
            value=5,
        )
        st.button("Press here to record", on_click=record, args=(mic, time))

        if "wf" in st.session_state:
            normalized_wf = normalize_wf(st.session_state.wf)

            st.pyplot(plot_wf(normalized_wf))

            st.session_state.test_wf_tag_gen = [(normalized_wf, 11)]
            st.session_state.test_chks = list(
                threshold_chunker(
                    st.session_state.test_wf_tag_gen,
                )
            )
            st.session_state.test_fvs = list(
                threshold_featurizer(st.session_state.test_chks)
            )

            st.pyplot(
                barplot_thresh(st.session_state.test_fvs, st.session_state.thresh)
            )
            test_thresh_df = mk_thresh_df(
                st.session_state.test_wf_tag_gen,
                st.session_state.thresh,
            )
            write_intervals(test_thresh_df, st.session_state.thresh)

            container = st.beta_container()
            st.session_state.thresh_adjust = container.number_input(
                "If you would like to adjust the thresh for this test, enter the amount here"
            )
            container.button(
                "See the change", on_click=adjust_thresh, args=(container,)
            )

            st.button(
                "Click here to run the model on your recording!", on_click=run_model
            )

            st.button("Clear current recording", on_click=clear_recording)

        if "numbers" in st.session_state:
            st.pyplot(
                barplot_colored(st.session_state.test_fvs, st.session_state.results)
            )
            st.pyplot(mk_event_location_plot(st.session_state.results))
            st.success(
                "The detected phone digits are "
                + "".join([str(num) for num in st.session_state.numbers])
            )

    if st.session_state.live == "Live":
        st.session_state.live_wf = LiveWf(mic)
        st.session_state.live_wf.start()

        st.session_state.pressed = st.button("Press here to start!")

        placeholder = st.empty()

        st.button("Press here to stop!", on_click=stop)

        store_in_ss("chk_stop", 8192)
        store_in_ss(
            "chunker",
            partial(fixed_step_chunker, chk_size=DFLT_CHK_SIZE, chk_step=DFLT_CHK_STEP),
        )

        while st.session_state.pressed:
            (
                st.session_state.bar_plt,
                st.session_state.results,
                st.session_state.number,
                st.session_state.thresh_fvs,
            ) = charts_for_live(
                st.session_state.live_wf,
                st.session_state.chk_stop,
                st.session_state.chunker,
                st.session_state.thresh,
                st.session_state.featurizer,
                st.session_state.model,
            )

            with placeholder.beta_container():
                st.pyplot(st.session_state.bar_plt, clear_figure=False)
                st.write("".join([str(num) for num in st.session_state.number]))

            st.session_state.chk_stop += 8192

        st.session_state.live_wf.stop()

        if "number" in st.session_state and not st.session_state.pressed:
            with placeholder.beta_container():
                st.pyplot(
                    barplot_colored(
                        st.session_state.thresh_fvs,
                        st.session_state.results,
                    )
                )
                # st.pyplot(mk_event_location_plot(st.session_state.results))
                st.success(
                    "The detected phone digits are "
                    + "".join([str(num) for num in st.session_state.number])
                )
