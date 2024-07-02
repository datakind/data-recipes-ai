from call_assistant import run_chainlit_mock

#
# Note, you can also debug by running directly, eg ...
#
# python3 call_assistant.py --chat_history '[{"author": "user","content": "How many rows does the population table have for Nigeria"}]'
#
# But this will hang. Below use the wrapper to terminate the process.
#


def main():
    """
    This function is the entry point of the program.
    It demonstrates different scenarios by calling the `run_chainlit_mock` function with different inputs.
    """

    # Assistant smalltalk
    run_chainlit_mock('[{"author": "user","content": "Hi"}]')

    # Memories, text output
    run_chainlit_mock(
        '[{"author": "user","content": "what is the population of Mali?"}]'
    )

    # Memories, image output
    run_chainlit_mock(
        '[{"author": "user","content": "plot a line chart of fatalities by month for Chad using HDX data as an image"}]'
    )

    # Recipe run, image output
    run_chainlit_mock(
        '[{"author": "user","content": "plot population pyramids for Nigeria"}]'
    )

    # Assistant on-the-fly SQL analysis of DB, text output
    run_chainlit_mock(
        '[{"author": "user","content": "How many rows does the population table have for Nigeria"}]'
    )

    # Assistant simple analysis and code interpretor, image output
    run_chainlit_mock('[{"author": "user","content": "Plot f{x}=10"}]')


if __name__ == "__main__":
    main()
