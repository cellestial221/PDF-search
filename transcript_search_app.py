import os
import re
import sys
import tempfile
from collections import defaultdict, OrderedDict

import streamlit as st

# Set page config as the very first Streamlit command
st.set_page_config(
    page_title="PDF Transcript Search",
    page_icon="🔍",
    layout="wide"
)

# Import pdfplumber with better error handling
try:
    import pdfplumber
except ImportError:
    st.error("This application requires pdfplumber. Please install it with: pip install pdfplumber")
    st.write("Checking your Python environment...")
    st.write(f"Python path: {sys.executable}")
    st.write(f"Python version: {sys.version}")
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
    st.code(result.stdout)
    st.stop()


class PDFTranscriptSearcher:
    """Class to handle searching through PDF transcript files."""

    def __init__(self, pdf_path, context_responses=2):
        """
        Initialise the PDFTranscriptSearcher.

        Args:
            pdf_path: Path to the PDF transcript file
            context_responses: Number of responses to include before and after a match
        """
        self.pdf_path = pdf_path
        self.context_responses = context_responses
        self.pages_content = {}  # Dictionary of {page_number: correctly_ordered_text}
        self.all_responses = []  # All responses across all pages in order
        self.response_page_map = {}  # Maps response index to page number
        self.response_internal_page_map = {}  # Maps response index to internal page number
        self.internal_page_numbers = {}  # Dictionary of {page_number: {left/right: number}}
        self.pages_responses = {}  # Dictionary of {page_number: list_of_responses}
        self.index_page = None  # The page where the index starts
        self._load_pdf()

    def _extract_raw_text_from_page(self, page):
        """Extract raw text from a PDF page using pdfplumber's extract_text method."""
        return page.extract_text() or ""

    def _clean_transcript_text(self, text):
        """
        Clean transcript text by removing irrelevant elements:
        - Standalone page numbers
        - Headers and footers
        - Extra whitespace
        """
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Skip standalone page numbers
            if re.match(r'^\s*\d+\s*$', line):
                continue

            # Skip header/footer lines
            if re.search(r'UK Covid-19 Inquiry', line) or re.search(r'\d+ March 2025', line):
                continue

            # Skip lines with "Pages" references
            if re.search(r'\(\d+\) Pages \d+ - \d+', line):
                continue

            # Add the cleaned line
            cleaned_lines.append(line.strip())

        return '\n'.join(cleaned_lines)

    def _identify_and_join_responses(self, text):
        """
        Identify complete responses and join lines within each response.
        Format Q. and A. as Q: and A: for better readability.

        Args:
            text: Text to process

        Returns:
            List of complete responses
        """
        lines = text.split('\n')
        responses = []
        current_response = []
        current_speaker = None

        # Regular expressions to identify speakers and response patterns
        speaker_pattern = re.compile(r'^(MR|MS|MRS|DR|PROFESSOR|LADY|LORD|SIR|THE WITNESS)\s+[A-Z]+[^:]*:', re.IGNORECASE)
        qa_pattern = re.compile(r'^([QA])\.\s', re.IGNORECASE)

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Check if this line starts a new response
            speaker_match = speaker_pattern.match(line)
            qa_match = qa_pattern.match(line)

            if speaker_match or qa_match:
                # Save the previous response if there is one
                if current_response:
                    response_text = ' '.join(current_response)
                    # Format Q. and A. as Q: and A:
                    response_text = re.sub(r'^Q\.\s+', 'Q: ', response_text)
                    response_text = re.sub(r'^A\.\s+', 'A: ', response_text)
                    responses.append(response_text)

                # Start a new response
                current_response = [line.strip()]
                if speaker_match:
                    current_speaker = speaker_match.group(0)
                else:
                    current_speaker = qa_match.group(0)
            else:
                # Check if this is a continuation of a response or a new response without a clear marker
                if current_response and not line.startswith(('MR ', 'MS ', 'MRS ', 'DR ', 'PROFESSOR ', 'LADY ', 'LORD ', 'SIR ', 'THE WITNESS')):
                    # This is a continuation of the current response
                    current_response.append(line.strip())
                else:
                    # Save the previous response if there is one
                    if current_response:
                        response_text = ' '.join(current_response)
                        # Format Q. and A. as Q: and A:
                        response_text = re.sub(r'^Q\.\s+', 'Q: ', response_text)
                        response_text = re.sub(r'^A\.\s+', 'A: ', response_text)
                        responses.append(response_text)

                    # Start a new response without a clear marker
                    current_response = [line.strip()]
                    current_speaker = None

        # Add the last response if there is one
        if current_response:
            response_text = ' '.join(current_response)
            # Format Q. and A. as Q: and A:
            response_text = re.sub(r'^Q\.\s+', 'Q: ', response_text)
            response_text = re.sub(r'^A\.\s+', 'A: ', response_text)
            responses.append(response_text)

        return responses

    def _process_court_transcript(self, page, page_num):
        """
        Process a court transcript page with multiple columns.
        Extract internal page numbers and properly organize text.

        Args:
            page: pdfplumber page object
            page_num: The page number

        Returns:
            Tuple of (processed_text, internal_page_numbers)
        """
        # Extract all words with positions
        words = page.extract_words(x_tolerance=3, y_tolerance=3)

        if not words:
            return self._clean_transcript_text(self._extract_raw_text_from_page(page)), None

        # Determine the page width and height
        page_width = page.width
        page_height = page.height
        mid_point = page_width / 2

        # Separate left and right columns based on x position
        left_words = [word for word in words if word['x0'] < mid_point]
        right_words = [word for word in words if word['x0'] >= mid_point]

        # Extract the actual text for processing
        left_words.sort(key=lambda w: (w['top'], w['x0']))
        right_words.sort(key=lambda w: (w['top'], w['x0']))

        # Group words by line (based on y-position)
        left_lines = self._group_words_by_line(left_words)
        right_lines = self._group_words_by_line(right_words)

        # Process left and right sides to remove line numbers and format Q/A
        left_text = self._process_transcript_lines(left_lines)
        right_text = self._process_transcript_lines(right_lines)

        # Extract internal page numbers before cleaning the text
        # They are typically displayed at the very bottom center of each column
        internal_page_numbers = {}

        # For UK Covid-19 Inquiry transcripts, look for the page number that appears centered
        # at the bottom of each column (e.g., "97", "98", "99", "100")

        # More precise method: Check the very last line in each column
        if left_lines:
            # Get the last 1-2 lines
            last_left_lines = left_lines[-2:] if len(left_lines) >= 2 else left_lines[-1:]
            for line in last_left_lines:
                line_text = ' '.join(word['text'] for word in line)
                # Look for a standalone number matching transcript page pattern
                if re.match(r'^\s*\d+\s*$', line_text):
                    try:
                        internal_page_numbers['left'] = int(line_text.strip())
                        # Once found, no need to look further
                        break
                    except ValueError:
                        pass

        if right_lines:
            # Get the last 1-2 lines
            last_right_lines = right_lines[-2:] if len(right_lines) >= 2 else right_lines[-1:]
            for line in last_right_lines:
                line_text = ' '.join(word['text'] for word in line)
                # Look for a standalone number matching transcript page pattern
                if re.match(r'^\s*\d+\s*$', line_text):
                    try:
                        internal_page_numbers['right'] = int(line_text.strip())
                        # Once found, no need to look further
                        break
                    except ValueError:
                        pass

        # If the above didn't work, try the most common pattern:
        # Look at the very bottom for standalone page numbers
        if not internal_page_numbers:
            bottom_threshold = page_height * 0.95  # Focus on the very bottom 5% of the page

            # For left column
            bottom_left_words = [w for w in left_words if w['top'] > bottom_threshold]
            bottom_left_words.sort(key=lambda w: -w['top'])  # Sort from bottom up

            for word in bottom_left_words:
                # Look for numbers that appear in isolation, are greater than 1-2 digits (to avoid line numbers)
                # and are placed at the very bottom
                if re.match(r'^\d{2,3}$', word['text']):  # 2-3 digit numbers (typical page numbers)
                    try:
                        internal_page_numbers['left'] = int(word['text'])
                        break
                    except ValueError:
                        pass

            # For right column
            bottom_right_words = [w for w in right_words if w['top'] > bottom_threshold]
            bottom_right_words.sort(key=lambda w: -w['top'])  # Sort from bottom up

            for word in bottom_right_words:
                if re.match(r'^\d{2,3}$', word['text']):
                    try:
                        internal_page_numbers['right'] = int(word['text'])
                        break
                    except ValueError:
                        pass

        # Combined text with left column first, then right column
        combined_text = left_text + "\n" + right_text

        # Clean the text to remove page numbers, headers, etc.
        return self._clean_transcript_text(combined_text), internal_page_numbers


    def _group_words_by_line(self, words):
        """Group words into lines based on y-position."""
        if not words:
            return []

        # Use a tolerance for grouping words on the same line
        line_tolerance = 3

        lines = []
        current_line = [words[0]]

        for word in words[1:]:
            # If this word is on the same line as the previous one
            if abs(word['top'] - current_line[0]['top']) <= line_tolerance:
                current_line.append(word)
            else:
                # Sort words in line by x-position
                current_line.sort(key=lambda w: w['x0'])
                lines.append(current_line)
                current_line = [word]

        # Add the last line
        if current_line:
            current_line.sort(key=lambda w: w['x0'])
            lines.append(current_line)

        return lines

    def _process_transcript_lines(self, lines):
        """Process grouped lines to remove line numbers and format Q/A patterns."""
        processed_lines = []

        for line in lines:
            # Skip empty lines
            if not line:
                continue

            # Convert words to text
            line_text = ' '.join(word['text'] for word in line)

            # Try to detect and remove line numbers at the beginning
            line_text = re.sub(r'^\s*\d+\s+', '', line_text)

            # Add the processed line
            if line_text.strip():
                processed_lines.append(line_text.strip())

        return '\n'.join(processed_lines)

    def _organize_responses_across_pages(self):
        """
        Organize all responses from all pages into a single sequence,
        keeping track of which page each response came from and its internal page number.
        This allows for handling excerpts that cross page boundaries.
        """
        all_responses = []
        response_page_map = {}
        response_internal_page_map = {}

        for page_num in sorted(self.pages_content.keys()):
            page_responses = self.pages_responses.get(page_num, [])
            internal_page_nums = self.internal_page_numbers.get(page_num, {})

            # For transcript pages, both left and right columns often have the same page number
            # Check if both columns have the same number (common in transcripts)
            if 'left' in internal_page_nums and 'right' in internal_page_nums:
                if internal_page_nums['left'] == internal_page_nums['right']:
                    # In this case, use the same page number for all responses on this page
                    internal_page = internal_page_nums['left']

                    for response in page_responses:
                        response_idx = len(all_responses)
                        all_responses.append(response)
                        response_page_map[response_idx] = page_num
                        response_internal_page_map[response_idx] = internal_page

                # If left and right have different numbers, use the appropriate one based on position
                else:
                    half_idx = len(page_responses) // 2

                    for idx, response in enumerate(page_responses):
                        response_idx = len(all_responses)
                        all_responses.append(response)
                        response_page_map[response_idx] = page_num

                        # Assign internal page number based on position in the page
                        # First half from left column, second half from right column
                        if idx < half_idx:
                            response_internal_page_map[response_idx] = internal_page_nums['left']
                        else:
                            response_internal_page_map[response_idx] = internal_page_nums['right']

            # If only one column has a page number, use it for all responses on this page
            elif 'left' in internal_page_nums:
                internal_page = internal_page_nums['left']

                for response in page_responses:
                    response_idx = len(all_responses)
                    all_responses.append(response)
                    response_page_map[response_idx] = page_num
                    response_internal_page_map[response_idx] = internal_page

            elif 'right' in internal_page_nums:
                internal_page = internal_page_nums['right']

                for response in page_responses:
                    response_idx = len(all_responses)
                    all_responses.append(response)
                    response_page_map[response_idx] = page_num
                    response_internal_page_map[response_idx] = internal_page

            # If no internal page numbers found, just map responses to the PDF page number
            else:
                for response in page_responses:
                    response_idx = len(all_responses)
                    all_responses.append(response)
                    response_page_map[response_idx] = page_num

        self.all_responses = all_responses
        self.response_page_map = response_page_map
        self.response_internal_page_map = response_internal_page_map

    def _load_pdf(self):
        """Load the PDF file and extract text with proper column handling."""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # Create a progress bar for PDF loading in Streamlit
                progress_bar = st.progress(0)
                progress_text = st.empty()

                total_pages = len(pdf.pages)

                # Dictionary to store responses by page
                self.pages_responses = {}

                for i, page in enumerate(pdf.pages, 1):
                    # Update progress
                    progress_percent = int((i / total_pages) * 100)
                    progress_bar.progress(progress_percent / 100)
                    progress_text.text(f"Processing page {i} of {total_pages}...")

                    # Extract raw text to check for index marker
                    raw_text = page.extract_text() or ""

                    # Check if this page contains "I N D E X" with spaces
                    if self.index_page is None and re.search(r'I\s+N\s+D\s+E\s+X', raw_text):
                        self.index_page = i
                        st.info(f"Index page detected: Page {i}. Excluding this page and all following pages.")

                    # Skip processing if this is part of the index
                    if self.index_page is not None and i >= self.index_page:
                        continue

                    # Process the page text to handle court transcript format
                    processed_text, internal_page_nums = self._process_court_transcript(page, i)

                    # Store the processed text
                    self.pages_content[i] = processed_text

                    # Store internal page numbers
                    if internal_page_nums:
                        self.internal_page_numbers[i] = internal_page_nums

                    # Process into responses
                    responses = self._identify_and_join_responses(processed_text)
                    self.pages_responses[i] = responses

                # Organize responses across all pages
                self._organize_responses_across_pages()

                # Clear the progress indicators
                progress_bar.empty()
                progress_text.empty()

                if not self.pages_content:
                    st.error("No text could be extracted from the PDF.")
                    st.stop()

        except Exception as e:
            st.error(f"Error loading PDF file: {str(e)}")
            st.exception(e)  # This will show the full traceback
            st.stop()

    def search_terms(self, terms):
        """
        Search the transcript for all specified terms, excluding index pages.
        Handle cross-page excerpts properly.

        Args:
            terms: List of search terms

        Returns:
            Dictionary mapping each term to a list of (excerpt, page_info) tuples
        """
        results = defaultdict(list)

        # Create a progress bar for search
        progress_bar = st.progress(0)
        progress_text = st.empty()

        total_terms = len(terms)
        for term_idx, term in enumerate(terms):
            # Update progress
            progress_percent = int((term_idx / total_terms) * 100)
            progress_bar.progress(progress_percent / 100)
            progress_text.text(f"Searching for term {term_idx+1} of {total_terms}: '{term}'...")

            # Case-insensitive search
            term_pattern = re.compile(fr'\b{re.escape(term)}\b', re.IGNORECASE)

            # Search through all responses (across all pages)
            for i, response in enumerate(self.all_responses):
                for match in term_pattern.finditer(response):
                    # Get the page number for this response
                    page_num = self.response_page_map.get(i, None)
                    if page_num is None:
                        continue

                    # Determine the range of responses to include
                    start_idx = max(0, i - self.context_responses)
                    end_idx = min(len(self.all_responses) - 1, i + self.context_responses)

                    # Extract context responses
                    context_responses = self.all_responses[start_idx:end_idx + 1]

                    # Get page numbers for all context responses
                    context_pages = [self.response_page_map.get(idx, None)
                                     for idx in range(start_idx, end_idx + 1)]

                    # Get internal page numbers for all context responses
                    context_internal_pages = [self.response_internal_page_map.get(idx, None)
                                             for idx in range(start_idx, end_idx + 1)]

                    # Filter out None values
                    context_pages = [p for p in context_pages if p is not None]
                    context_internal_pages = [p for p in context_internal_pages if p is not None]

                    # Create page range info
                    if len(set(context_pages)) > 1:
                        page_info = f"Pages {min(context_pages)}-{max(context_pages)}"
                    else:
                        page_info = f"Page {page_num}"

                    # Add internal page numbers if available
                    if context_internal_pages:
                        if len(set(context_internal_pages)) > 1:
                            internal_page_info = f"Internal pages {min(context_internal_pages)}-{max(context_internal_pages)}"
                        else:
                            internal_page_info = f"Internal page {context_internal_pages[0]}"

                        page_info = f"{page_info} ({internal_page_info})"

                    # Highlight the matching term in the matched response
                    matched_term = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()

                    # Create highlighted response with the exact match bolded
                    highlighted_response = (
                        response[:start_pos] +
                        f"**{matched_term}**" +
                        response[end_pos:]
                    )

                    # Replace the matched response with the highlighted version
                    context_responses[i - start_idx] = highlighted_response

                    # Join the context responses with newlines
                    excerpt = '\n\n'.join(context_responses)

                    results[term].append((excerpt, page_info))

        # Clear the progress indicators
        progress_bar.empty()
        progress_text.empty()

        return results


def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary location and return the path."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    # Write the file to the temporary directory
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return temp_path


def main():
    """Main Streamlit app function."""
    st.title("PDF Transcript Search App")
    st.markdown("""
    This app allows you to search PDF transcript files for specific terms and display the results
    with complete responses. It's designed specifically for court-style transcripts with Q&A format.
    """)

    # Sidebar for inputs
    with st.sidebar:
        st.header("Upload and Search")

        # File upload
        uploaded_file = st.file_uploader("Upload transcript PDF", type="pdf")

        # Search terms inputs
        st.subheader("Search Terms")
        search_terms_input = st.text_area(
            "Enter search terms (one per line)",
            height=150,
            help="Enter each search term on a new line. The search is case-insensitive."
        )

        # Context setting
        context_responses = st.slider(
            "Number of responses to include as context",
            1, 5, 2,
            help="How many responses to show before and after the matching response"
        )

        # Search button
        search_button = st.button("Search", type="primary")

    # Main content area
    if uploaded_file is not None:
        # Display file information
        st.subheader("File Information")
        st.write(f"Filename: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size / 1024:.1f} KB")

        # Save uploaded file to temp location
        temp_file_path = save_uploaded_file(uploaded_file)

        if search_button:
            # Get search terms
            search_terms = [term.strip() for term in search_terms_input.splitlines() if term.strip()]

            if search_terms:
                with st.spinner("Processing PDF and searching for terms..."):
                    # Perform search
                    searcher = PDFTranscriptSearcher(temp_file_path, context_responses)
                    results = searcher.search_terms(search_terms)

                # Display results
                st.header("Search Results")

                total_results = sum(len(matches) for matches in results.values())
                st.markdown(f"**Total matches found: {total_results}**")

                if searcher.index_page is not None:
                    st.info(f"Index detected on page {searcher.index_page}. Results exclude this and all following pages.")

                # Create tabs for each search term
                if search_terms:
                    tabs = st.tabs(search_terms)

                    for i, term in enumerate(search_terms):
                        term_results = results.get(term, [])

                        with tabs[i]:
                            st.markdown(f"### Results for '{term}' ({len(term_results)} matches)")

                            if not term_results:
                                st.write("No matches found.")
                            else:
                                for j, (excerpt, page_info) in enumerate(term_results, 1):
                                    with st.expander(f"Match {j} ({page_info})"):
                                        # Use unsafe_allow_html=True to ensure Markdown formatting works
                                        st.markdown(excerpt, unsafe_allow_html=True)
            else:
                st.warning("Please enter at least one search term.")
    else:
        st.info("Please upload a transcript PDF file to begin.")

        # Sample instructions
        with st.expander("See Instructions"):
            st.markdown("""
            ### How to use this app:

            1. **Upload a PDF**: Use the sidebar to upload your transcript PDF file.
            2. **Enter search terms**: Add one search term per line in the text area.
            3. **Adjust context**: Use the slider to set how many responses to show before and after matches.
            4. **Search**: Click the Search button to begin processing.
            5. **Review results**: Results are organized by search term and can be expanded to view the context.

            ### Tips:

            - The app works best with court-style transcripts that have a clear Q&A format
            - It automatically detects and excludes index pages
            - Search is case-insensitive
            - For multiword phrases, simply type them normally (e.g., "Project Defend")
            """)


if __name__ == "__main__":
    main()
