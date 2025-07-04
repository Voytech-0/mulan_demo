"""
Centralized settings configuration for the MULAN demo app.
All UI dimensions, heights, and styling parameters are defined here.
"""

# =============================================================================
# LAYOUT DIMENSIONS
# =============================================================================

# Container heights
IMAGE_DISPLAY_CONTAINER_HEIGHT = "56vh"
METADATA_DISPLAY_HEIGHT = "16vh"

# Component heights
IMAGE_HEIGHT = "25vh"
IMAGE_MAX_HEIGHT = "30vh"
IMAGE_MIN_HEIGHT = "4vh"
COORDINATES_HEIGHT = "16vh"
METADATA_HEIGHT = "4vh"
NO_IMAGE_HEIGHT = "4vh"
GENERATIVE_PLACEHOLDER_HEIGHT = "30vh"

# =============================================================================
# STYLING CONSTANTS
# =============================================================================

# Colors
BORDER_COLOR = "#dee2e6"
BACKGROUND_COLOR = "#2c3e50"
PLACEHOLDER_BACKGROUND = "#2c3e50"
PLACEHOLDER_BORDER = "#666"
TEXT_COLOR = "#666"

# Spacing
STANDARD_PADDING = "0.5rem"
STANDARD_MARGIN = "0.5rem"
IMAGE_PADDING = "0.5rem"

# =============================================================================
# COMPONENT STYLES
# =============================================================================

# Image styles
SELECTED_IMAGE_STYLE = {
    'max-width': '100%', 
    'height': IMAGE_HEIGHT,
    'max-height': IMAGE_MAX_HEIGHT,
    'object-fit': 'contain', 
    'display': 'none',
    'padding': IMAGE_PADDING
}

NO_IMAGE_MESSAGE_STYLE = {
    'display': 'none', 
    'text-align': 'center', 
    'padding': STANDARD_PADDING, 
    'height': NO_IMAGE_HEIGHT
}

NO_METADATA_MESSAGE_STYLE = {
    'display': 'none', 
    'text-align': 'center', 
    'padding': STANDARD_PADDING, 
    'height': NO_IMAGE_HEIGHT
}

GENERATIVE_PLACEHOLDER_STYLE = {
    'display': 'none', 
    'text-align': 'center', 
    'padding': STANDARD_PADDING, 
    'height': GENERATIVE_PLACEHOLDER_HEIGHT, 
    'border': f'2px dashed {PLACEHOLDER_BORDER}', 
    'border-radius': '5px', 
    'background-color': PLACEHOLDER_BACKGROUND
}

# Container styles
COORDINATES_DISPLAY_STYLE = {
    'height': COORDINATES_HEIGHT, 
    'overflow-y': 'auto', 
    'border': f'1px solid {BORDER_COLOR}', 
    'padding': STANDARD_PADDING, 
    'width': '100%',
    'box-sizing': 'border-box'
}

METADATA_DISPLAY_STYLE = {
    'height': METADATA_DISPLAY_HEIGHT, 
    'overflow-y': 'auto', 
    'border': f'1px solid {BORDER_COLOR}', 
    'padding': STANDARD_PADDING,
    'width': '100%',
    'box-sizing': 'border-box'
}

# Table styles
TABLE_STYLE = {
    'width': '100%', 
    'border-collapse': 'collapse'
}

CELL_STYLE = {
    'text-align': 'left', 
    'padding': '8px'
}

CELL_STYLE_RIGHT = {
    'text-align': 'right', 
    'padding': '8px'
}

# Message styles
EMPTY_METADATA_STYLE = {
    'text-align': 'center', 
    'color': TEXT_COLOR, 
    'padding': '1rem'
}

# =============================================================================
# FIGURE SETTINGS
# =============================================================================

IMAGE_FIGURE_SIZE = (6, 6)

# =============================================================================
# STYLE HELPER FUNCTIONS
# =============================================================================

def get_image_style(display_type='block'):
    """Get image style with specified display type."""
    style = SELECTED_IMAGE_STYLE.copy()
    style['display'] = display_type
    return style

def get_generative_placeholder_style(display_type='block'):
    """Get generative placeholder style with specified display type."""
    style = GENERATIVE_PLACEHOLDER_STYLE.copy()
    style['display'] = display_type
    return style

def get_no_image_message_style(display_type='block'):
    """Get no image message style with specified display type."""
    style = NO_IMAGE_MESSAGE_STYLE.copy()
    style['display'] = display_type
    return style 

# Function to get no image message style for callbacks
def get_no_metadata_message_style(display_type='block'):
    """Get no image message style with specified display type."""
    style = NO_IMAGE_MESSAGE_STYLE.copy()
    style['display'] = display_type
    return style 