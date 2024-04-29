from pathlib import Path
import streamlit as st
import wikipedia
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.agent import ReActAgent
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import TextNode
from llama_index.core.tools import RetrieverTool, ToolMetadata, FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader
from pydantic import BaseModel

system_prompt = "You are a movie recommender system."

llm_4_turbo = OpenAI(
    model="gpt-4-turbo-preview", temperature=0, system_prompt=system_prompt, timeout=120
)
llm_3_turbo = OpenAI(
    model="gpt-3.5-turbo-preview",
    temperature=0,
    system_prompt=system_prompt,
    timeout=120,
)

genres = {
    "fantasy": {
        "wiki_pages": ["Fantasy film"],
        "web_pages": [
            "https://www.imdb.com/search/title/?title_type=feature&genres=fantasy",
            "https://editorial.rottentomatoes.com/guide/best-fantasy-movies-of-all-time",
        ],
    },
    "comedy": {
        "wiki_pages": ["Comedy film"],
        "web_pages": [
            "https://www.imdb.com/search/title/?genres=comedy&explore=title_type%2Cgenres",
            "https://editorial.rottentomatoes.com/guide/essential-comedy-movies",
        ],
    },
    "documentary": {
        "wiki_pages": ["Documentary film"],
        "web_pages": [
            "https://www.imdb.com/search/title/?genres=documentary",
            "https://topdocumentaryfilms.com",
        ],
    },
}

viewer_types = ["IT geek", "average human", "social life enjoyer"]

social_settings = ["alone", "with friends", "with family", "with partner"]

platforms = ["Netflix", "HBO", "Amazon Prime", "Disney+", "Apple TV"]

movie_types = ["movie", "series", "multi-part movie"]

already_seen_rating = {
    "The Lord of the Rings: The Fellowship of the Ring": 10,
    "The Lord of the Rings: The Two Towers": 9,
    "The Lord of the Rings: The Return of the King": 8,
    "Inception": 6,
    "The Matrix": 9,
    "The Hobbit: An Unexpected Journey": 8.5,
    "The Dark Knight": 7,
    "Interstellar": 4.5,
    "Conan the Barbarian": 3.5,
}


class Movie(BaseModel):
    """
    Movie that can be recommended to the user. Name must be exact name of a movie. For example The Lord of the Rings: The Fellowship of the Ring.
    Description should be a long text describing the movie.
    """

    name: str
    description: str


class MoviesByGenre(BaseModel):
    """List of movies of a given genre"""

    movies: list[Movie]


class MovieDisplay(BaseModel):
    """
    Movie display for the user. Name should be exact name of a movie. For example The Lord of the Rings: The Fellowship of the Ring.
    Name and description should contain emojis.
    """

    name: str
    description: str


class MovieList(BaseModel):
    """List of movies for the user to watch."""

    movies: list[MovieDisplay]


def gen_movies(text: str) -> list[Movie]:
    summarizer = TreeSummarize(output_cls=MoviesByGenre, llm=llm_4_turbo)
    movies_by_genre = summarizer.get_response(
        "Recommend separate movies a person can watch.", [text]
    )

    return movies_by_genre.movies


def get_movies(genre: str):
    all_movies = []

    wiki_pages = genres[genre]["wiki_pages"]
    web_pages = genres[genre]["web_pages"]

    for wiki_page in wiki_pages:
        print(f"Processing Wiki {wiki_page}")
        wiki_page_content = wikipedia.page(wiki_page).content
        all_movies += gen_movies(wiki_page_content)

    for web_page in web_pages:
        print(f"Processing Web {web_page}")
        web_page_content = (
            SimpleWebPageReader(html_to_text=True).load_data([web_page])[0].text
        )
        all_movies += gen_movies(web_page_content)

    return all_movies


def generate_recommendation(
    genre: str,
    viewer_type: str,
    social_settings: str,
    movie_type: str,
    already_seen_ratings: dict,
):  
    print(genre, viewer_type)

    index_dir_name = Path(f"./index_{genre.lower()}/")
    if index_dir_name.exists():
        storage_context = StorageContext.from_defaults(persist_dir=index_dir_name)

        movies_index = load_index_from_storage(storage_context)
    else:
        all_movies = get_movies(genre)

        movies_nodes = [
            TextNode(text=f"{p.name} - {p.description}") for p in all_movies
        ]
        movies_index = VectorStoreIndex(movies_nodes)

        movies_index.storage_context.persist(persist_dir=index_dir_name)

    # def load_wikipedia_details(movie: str) -> bool:
    #     """
    #     Loads additional information from Wikipedia
    #     """
    #     pass

    # def check_platform(movie: str, plarform: str) -> bool:
    #     """
    #     Checks if the movie is available on the platform
    #     """
    #     pass

    query_engine_tools = [
        RetrieverTool(
            movies_index.as_retriever(similarity_top_k=20),
            metadata=ToolMetadata(
                name="movies_list",
                description=("Retrieves movies from the index based on the query."),
            ),
        ),
    ]

    # query_engine_tools += [FunctionTool.from_defaults(fn=load_wikipedia_details)]
    # query_engine_tools += [FunctionTool.from_defaults(fn=check_platform)]

    agent_prompt = f"""
    Generate complete movie recommendation prepared for a {viewer_type} of genre {genre}.
    The film should be suitable for social setting {social_settings}. It should be a {movie_type}.
    Check the {already_seen_ratings} for the movies already seen by the viewer and use the ratings to recommend new movies.
    The rating in the dictionary is out of 10. 0 being the lowest and 10 being the highest.
    Don't recommend movies that have already been seen by the viewer.
    Divide each movie into parts and use tools for each of them separately.
    For each movie list name, description and year. Use emojis in description and name.
    Describe why you think the viewer would enjoy the movie.
    Try to recommend 3 movies.
    """

    agent = ReActAgent.from_tools(
        query_engine_tools, llm=llm_4_turbo, verbose=True, max_iterations=20
    )
    agent_response = agent.chat(agent_prompt).response

    summarizer = TreeSummarize(output_cls=MovieList, llm=llm_4_turbo)
    response = summarizer.get_response("Parse movie list", [agent_response])

    print(agent_response)

    return response.movies


def main():
    st.title("Movie Recommendation System")

    genre = st.selectbox("Choose a genre of movies:", genres.keys(), index=0)

    viewer_type = st.radio("Who is watching?", viewer_types)

    social_setting = st.radio("Social setting:", social_settings)

    movie_type = st.radio("Type of movie:", movie_types)

    # platform = st.radio("Platform:", platforms)

    if st.button("Recommend Movies"):
        movie_list = generate_recommendation(
            genre, viewer_type, social_setting, movie_type, already_seen_rating
        )

        st.subheader(
            f"Movies Recommended for *{viewer_type}* of genre *{genre}* ",
            divider="rainbow",
        )
        for item in movie_list:
            st.text_area(f"**{item.name}**", item.description)


if __name__ == "__main__":
    # generate_recommendation("fantasy", "IT geek", "with partner", "movie", already_seen_rating)
    main()
