from .config_css import CSS as CONFIG_CSS


CSS = """
    Screen {
        background: #1a1b26;  /* Dark blue-grey background */
    }

    Container {
        layout: vertical;
        height: 100%;
        border: double rgb(91, 164, 91);
    }

    Horizontal {
        width: 100%;
    }

    #middle-pane {
        width: 100%;
        height: 1fr;
        border: round rgb(205, 133, 63);
        content-align: center middle;
    }

    #lower-middle-pane {
        width: 100%;
        height: 1fr;  
        border: round rgb(205, 133, 63);
        content-align: center middle;
    }

    #bottom-pane {
        width: 100%;
        height: 2fr;
        border: round rgb(205, 133, 63);
    }

    #status-row {
        height: 3;
        width: 100%;
        layout: horizontal;
        content-align: center middle;
        margin: 1 1;
    }

    #session-row {
        height: 3;
        width: 100%;
        layout: horizontal;
        content-align: center middle;
        margin: 1 1;
    }

    #status-indicator {
        content-align: center middle;
        width: 1fr;
        height: 3;
        background: #2a2b36;
        border: solid rgb(91, 164, 91);
        padding: 0 1;
    }

    #send-button {
        width: 12;
        height: 3;
        margin-left: 1;
        background: #2a2b36;
        border: solid rgb(91, 164, 91);
    }

    #config-button {
        width: 12;
        height: 3;
        margin-left: 1;
        background: #2a2b36;
        border: solid rgb(91, 164, 91);
    }
    
    #quit-button {
        width: 12;
        height: 3;
        margin-left: 1;
        background: #2a2b36;
        border: solid rgb(91, 164, 91);
    }

    #session-display {
        height: 3;
        width: 1fr;
        content-align: center middle;
        background: #2a2b36;
        border: solid rgb(91, 164, 91);
        padding: 0 1;
    }

    #amp-graph {
        width: 24;
        height: 3;
        margin-left: 1;
        background: #2a2b36;
        border: solid rgb(91, 164, 91);
    }

    Static {
        color: white;
    }
    
""" + CONFIG_CSS
