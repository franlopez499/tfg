import streamlit as st
import pages.upscale
import pages.segmentation
import awesome_streamlit as ast
    #"Home": pages.home,
    #"About": src.pages.about,

PAGES = {
    "Upscale": pages.upscale,
    "Segmentation": pages.segmentation,
}
ast.core.services.other.set_logging_format()
def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)
 
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is maintained by Francisco López Toledo. You can learn more about me at
        [github.com](https://google.com).
"""
    )


if __name__ == "__main__":
    main()