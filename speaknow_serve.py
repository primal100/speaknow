from textual_serve.server import Server


def run():
    server = Server("python -m speaknow")
    server.serve()


if __name__ == "__main__":
    run()
