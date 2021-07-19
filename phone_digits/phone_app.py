"""
Streamlit app for phone_digit detection
"""

import streamlit as st
import numpy as np

from phone_digits_dacc import mk_dacc
from streamlit_controller import (
    store_in_ss,
    annotations,
    adjust_thresh,
    clear_recording,
    run_model,
    write_intervals,
    mk_featurizer_and_model,
    download_model,
)
from controller import (
    threshold_chunker,
    threshold_featurizer,
    MetricOptimizedThreshold,
    mk_thresh_df,
    mk_thresh_chks,
    normalize_wf_upload,
    plot_wf,
    barplot_thresh,
    barplot_colored,
    mk_event_location_plot,
    upload_model,
    upload_audio,
)

st.title("Phone Digits")

train = st.selectbox("Would you like to train a new model?", ["Yes", "No"])

if train == "Yes":
    st.session_state.zip_file = st.file_uploader(
        label="Please upload the phone_digits training data!"
    )

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
                "mot",
                MetricOptimizedThreshold().fit(
                    st.session_state.wfs_and_tags, st.session_state.annotations
                ),
            )
            store_in_ss(
                "thresh",
                st.session_state.mot.score(),
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
            col1, col2 = st.beta_columns(2)
            with col1:
                st.button(
                    "Click here to save your model",
                    on_click=download_model,
                    args=(st.session_state.model, "model", col2),
                )
                st.button(
                    "Click here to save your featurizer",
                    on_click=download_model,
                    args=(st.session_state.featurizer, "featurizer", col2),
                )
                st.button(
                    "Click here to save your metric optimized threshold",
                    on_click=download_model,
                    args=(st.session_state.mot, "mot", col2),
                )


if train == "No":
    model_pkl = st.file_uploader(
        label="Please upload your model.pkl file here", type="pkl"
    )
    featurizer_pkl = st.file_uploader(
        label="Please upload your featurizer.pkl file here", type="pkl"
    )
    mot_pkl = st.file_uploader(label="Please upload your mot.pkl file here", type="pkl")

    if model_pkl and featurizer_pkl and mot_pkl:
        st.session_state.model = upload_model(model_pkl)
        st.session_state.featurizer = upload_model(featurizer_pkl)
        try:
            st.session_state.thresh = upload_model(featurizer_pkl).score()
        except EOFError:
            st.session_state.thresh = 0.0037
        st.success(
            f"🎈 Done! Your featurizer, model, and metric optimized threshold have been successfully decoded!"
        )

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

    test_audio = st.file_uploader(
        label="Please upload an audio file you would like to test", type=['wav', 'aiff', 'flac']
    )

    if test_audio:
        st.session_state.wf, _ = upload_audio(test_audio)

    if "wf" in st.session_state:
        wf = np.float()
        normalized_wf = normalize_wf_upload(st.session_state.wf)

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

        st.pyplot(barplot_thresh(st.session_state.test_fvs, st.session_state.thresh))
        test_thresh_df = mk_thresh_df(
            st.session_state.test_wf_tag_gen,
            st.session_state.thresh,
        )
        write_intervals(test_thresh_df, st.session_state.thresh)

        container = st.beta_container()
        st.session_state.thresh_adjust = container.number_input(
            "If you would like to adjust the thresh for this test, enter the amount here"
        )
        container.button("See the change", on_click=adjust_thresh, args=(container,))

        st.button("Click here to run the model on your recording!", on_click=run_model)

        st.button("Clear current recording", on_click=clear_recording)

    if "numbers" in st.session_state:
        st.pyplot(barplot_colored(st.session_state.test_fvs, st.session_state.results))
        st.pyplot(mk_event_location_plot(st.session_state.results))
        st.success(
            "The detected phone digits are "
            + "".join([str(num) for num in st.session_state.numbers])
        )
