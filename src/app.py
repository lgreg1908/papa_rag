import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(page_title="RAG Document Explorer", layout="wide")
    st.title("RAG Document Explorer")

    # Initialize browsing state
    if 'browse_path' not in st.session_state:
        st.session_state.browse_path = os.getcwd()

    # Sidebar: folder browser
    with st.sidebar:
        st.header("Folder Browser")
        st.write(f"**Current:** {st.session_state.browse_path}")

        # Navigate up
        parent = os.path.dirname(st.session_state.browse_path)
        if parent and os.path.isdir(parent):
            if st.button("Up", key="up_button"):
                st.session_state.browse_path = parent

        # List and select subdirectories
        try:
            dirs = [d for d in os.listdir(st.session_state.browse_path)
                    if os.path.isdir(os.path.join(st.session_state.browse_path, d))]
        except PermissionError:
            dirs = []
        dirs.sort()
        selected = st.selectbox("Subdirectories", options=dirs, key="subdirs_select")
        if st.button("Enter", key="enter_button") and selected:
            new_path = os.path.join(st.session_state.browse_path, selected)
            if os.path.isdir(new_path):
                st.session_state.browse_path = new_path

        # Connect to the selected folder
        if st.button("Connect", key="connect_button"):
            folder = st.session_state.browse_path
            if folder and os.path.isdir(folder):
                st.success(f"✔ Connected to folder: {folder}")
                # TODO: hook into watcher.ingest_folder(folder)
            else:
                st.error("Invalid folder path selected.")

    # Main area
    st.write("Use the sidebar to browse and connect to a folder to begin ingestion.")

if __name__ == "__main__":
    main()
