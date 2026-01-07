CSS = """
#config-dialog {
    background: #1a1b26;
    border: thick rgb(91, 164, 91);
    height: 80vh;  /* Fixed height for the modal */
    width: 80%;
    margin: 2 4;
    padding: 0;   /* Remove padding here so header/footer hit the edges */
    layout: vertical;
}

/* Sticky Header */
#config-title {
    text-align: center;
    width: 100%;
    height: 3;
    background: rgb(91, 164, 91);
    color: white;
    text-style: bold;
    content-align: center middle;
}

/* Scrollable Body - Takes all remaining space */
#config-body {
    height: 1fr;
    overflow-y: scroll;
    padding: 1 2; /* Give the fields some breathing room */
}

/* Sticky Footer */
#config-buttons {
    height: 5;
    background: #16161e; /* Slightly darker to distinguish footer */
    border-top: solid rgb(91, 164, 91);
    content-align: center middle;
    dock: bottom; /* Forces it to stay at the bottom */
}

/* Row Styling */
#modalities-row, .input-row {
    height: auto;
    margin-bottom: 1;
}

.input-row Vertical {
    height: auto;
    width: 1fr;
}

/* Widget Styling */
Label {
    margin-top: 1;
    color: #9aa5ce;
    text-style: bold;
}

Input {
    margin-bottom: 1;
    border: round rgb(91, 164, 91);
}

Checkbox {
    width: auto;
}

/* Rectangular Buttons */
#save_config, #cancel_config {
    width: 20; 
    height: 3;
    border: none;
    text-style: bold;
    margin: 0 2;
}

/* Container for the three horizontal fields */
#model-settings-row {
    height: auto;
    width: 100%;
    margin-bottom: 1;
}

/* Ensure each vertical column takes up 1/3 of the width */
#model-settings-row Vertical {
    width: 1fr;
    height: auto;
    padding-right: 1; /* Space between columns */
}

/* Optional: Adjust Input height inside the row if they look cramped */
#model-settings-row Input {
    width: 100%;
}

#save_config { background: #31ad62; color: white; }
#cancel_config { background: #c94a5b; color: white; }
"""