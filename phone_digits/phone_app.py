"""
Streamlit app for phone digit detection
"""

import streamlit as st
import numpy as np

from taped import list_recording_device_index_names, LiveWf


from utils import upload_model, upload_audio
from controller import PhoneDigitsController, PhoneDigitsTestController
from streamlit_utils import (
    store_in_ss,
    annotations,
    show_adjustment,
    run_model,
    write_intervals,
    download_model,
    record,
    stop,
    clear_recording,
    display_results,
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
        store_in_ss("Controller", PhoneDigitsController(st.session_state.zip_file))
        if "random_chk_ids" not in st.session_state:
            (
                st.session_state.chks,
                st.session_state.chk_ids,
                st.session_state.random_chk_ids,
            ) = st.session_state.Controller.mk_annotations_data()
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
            if "mot_trained" not in st.session_state:
                st.session_state.Controller.train_thresh_model(
                    st.session_state.annotations
                )
                st.session_state.Controller.mk_train_thresh_df()
                st.session_state.Controller.mk_train_thresh_chks()
                st.session_state.mot_trained = True

            st.markdown("""---""")

            st.button(
                "Click here to build your featurizer",
                on_click=st.session_state.Controller.mk_featurizer,
                args=(featurizer_choice,),
            )
            st.button(
                "Click here to build your model",
                on_click=st.session_state.Controller.mk_model,
                args=(model_choice,),
            )

        if "Controller" in st.session_state:
            if hasattr(st.session_state.Controller, "featurizer") and hasattr(
                st.session_state.Controller, "model"
            ):
                st.markdown("""---""")

                st.success(
                    f"The model was able to correctly classify "
                    f"{100 * round(np.sum(st.session_state.Controller.scores == st.session_state.Controller.tags) / len(st.session_state.Controller.tags), 2)}% "
                    f"of the digits pressed in the training data"
                )

                st.write("""---""")
                col1, col2 = st.beta_columns(2)
                st.session_state.download_count = 3
                with col1:
                    st.button(
                        "Click here to save your featurizer",
                        on_click=download_model,
                        args=(
                            st.session_state.Controller.featurizer,
                            "featurizer",
                            col2,
                        ),
                    )
                    st.button(
                        "Click here to save your model",
                        on_click=download_model,
                        args=(st.session_state.Controller.model, "model", col2),
                    )
                    st.button(
                        "Click here to save your threshold",
                        on_click=download_model,
                        args=(st.session_state.Controller.score(), "threshold", col2),
                    )
                    st.write("""---""")

        if "download_count" in st.session_state:
            if st.session_state.download_count == 3:
                st.session_state.TestController = PhoneDigitsTestController(
                    *st.session_state.Controller.params_for_test()
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
        model = upload_model(model_pkl)
        featurizer = upload_model(featurizer_pkl)
        try:
            thresh = upload_model(featurizer_pkl)
        except EOFError:
            thresh = 0.0037
        st.session_state.TestController = PhoneDigitsTestController(
            featurizer, model, thresh
        )
        st.success(
            f"🎈 Done! Your featurizer, model, and metric optimized threshold have been successfully decoded!"
        )

if "TestController" in st.session_state:
    st.session_state.live = st.selectbox(
        "How would you like to test the model?",
        # ["Upload Recording", "Live Recording", "Livestream"],
        ["Upload Recording"],
    )

if "live" in st.session_state:
    if st.session_state.live in ["Live Recording", "Upload Recording"]:
        if st.session_state.live == "Live Recording":
            mic = st.selectbox(
                "Which microphone would you like to use?",
                list_recording_device_index_names(),
            )
            time = st.number_input(
                "How long would you like the recording to be?",
                min_value=0,
                max_value=15,
                value=5,
            )
            st.button("Press here to record", on_click=record, args=(mic, time))

            if "wf" in st.session_state:
                st.session_state.TestController.set_wf(
                    st.session_state.wf, upload=False
                )
        else:
            test_audio = st.file_uploader(
                label="Please upload an audio file you would like to test",
                type=["wav", "aiff", "flac"],
            )
            if test_audio:
                st.session_state.wf = upload_audio(test_audio)
                st.session_state.TestController.set_wf(st.session_state.wf, upload=True)

        if "wf" in st.session_state:
            st.pyplot(st.session_state.TestController.plot_wf())
            st.session_state.TestController.prep_for_test()
            st.pyplot(st.session_state.TestController.barplot_thresh())
            write_intervals(
                st.session_state.TestController.test_thresh_df,
                st.session_state.TestController.threshold,
            )
            container = st.beta_container()
            st.session_state.thresh_adjust = container.number_input(
                "If you would like to adjust the thresh for this test, enter the amount here",
            )
            container.button(
                "See the change", on_click=show_adjustment, args=(container,)
            )
            st.button(
                "Click here to run the model on your recording!",
                on_click=run_model,
            )
            st.button(
                "Clear current recording",
                on_click=clear_recording,
            )
        if "number" in st.session_state:
            display_results(final=True)

    else:
        st.session_state.live_wf = LiveWf()
        st.session_state.live_wf.start()

        st.session_state.pressed = st.button("Press here to start!")

        placeholder = st.empty()

        st.button("Press here to stop!", on_click=stop)

        store_in_ss("chk_stop", 44100)

        while st.session_state.pressed:
            (
                st.session_state.plt,
                st.session_state.results,
                st.session_state.number,
                st.session_state.thresh_fvs,
            ) = st.session_state.TestController.charts_for_live(
                st.session_state.live_wf, st.session_state.chk_stop
            )

            with placeholder.beta_container():
                display_results(final=False)

            st.session_state.chk_stop += 44100

        st.session_state.live_wf.stop()

        if "number" in st.session_state and not st.session_state.pressed:
            with placeholder.beta_container():
                display_results(final=True)
