"""ComfyUI-UseapiNet — Custom nodes for the Useapi.net API."""
try:
    from .useapi_nodes import NODE_CLASS_MAPPINGS as _CORE
    from .useapi_nodes import NODE_DISPLAY_NAME_MAPPINGS as _CORE_D
    from .useapi_extra import NODE_CLASS_MAPPINGS as _EXTRA
    from .useapi_extra import NODE_DISPLAY_NAME_MAPPINGS as _EXTRA_D
except ImportError:
    from useapi_nodes import NODE_CLASS_MAPPINGS as _CORE
    from useapi_nodes import NODE_DISPLAY_NAME_MAPPINGS as _CORE_D
    from useapi_extra import NODE_CLASS_MAPPINGS as _EXTRA
    from useapi_extra import NODE_DISPLAY_NAME_MAPPINGS as _EXTRA_D

NODE_CLASS_MAPPINGS = {**_CORE, **_EXTRA}
NODE_DISPLAY_NAME_MAPPINGS = {**_CORE_D, **_EXTRA_D}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
