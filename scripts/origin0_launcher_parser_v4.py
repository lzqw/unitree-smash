"""Register pinned Isaac launcher flags without consuming required user fields."""


def add_origin0_launcher_args(parser, launcher):
    required = [action for action in parser._actions if action.required]
    try:
        for action in required:
            action.required = False
        launcher.add_app_launcher_args(parser)
    finally:
        for action in required:
            action.required = True
