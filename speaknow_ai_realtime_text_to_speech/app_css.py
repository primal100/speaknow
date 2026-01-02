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

/* Shared style for all main app buttons */
    #send-button, #config-button, #quit-button {
        width: 16;             /* Fixed width for rectangular look */
        height: 3;
        margin-left: 1;
        border: none;          /* Remove the thin line border */
        text-style: bold;
        content-align: center middle;
    }

    /* Specific colors to match the professional TUI feel */
    #send-button {
        background: #31ad62;   /* Green */
        color: white;
    }

    #config-button {
        background: #414868;   /* Slate Blue */
        color: white;
    }

    #quit-button {
        background: #c94a5b;   /* Red */
        color: white;
    }

    /* Interactivity states */
/* Interactivity states */
    #send-button:hover, #config-button:hover, #quit-button:hover {
        tint: white 20%;  /* This makes the existing background look lighter */
    }

    #send-button:focus, #config-button:focus, #quit-button:focus {
        background: #bbbbbb; 
        color: black;
    }
    #version-display {
        height: 3;
        width: 0.5fr;
        content-align: left middle;
        background: #2a2b36;
        border: solid rgb(91, 164, 91);
        padding: 0 1;
    }
    
    #session-display {
        height: 3;
        width: 1.5fr;
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
