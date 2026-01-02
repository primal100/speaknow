from textual_serve.server import Server

if __name__ == "__main__":
    server = Server("python -m speaknow")
    server.serve()
