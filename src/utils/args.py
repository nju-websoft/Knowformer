def str2bool(v):
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type_, default, help_, **kwargs):
        type_ = str2bool if type_ == bool else type_
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type_,
            help=help_ + ' Default: %(default)s.',
            **kwargs)


def print_arguments(logger, args):
    logger.info('-----------  Configuration Arguments -----------')
    for arg, value in vars(args).items():
        logger.info('%s: %s' % (arg, value))
    logger.info('------------------------------------------------')
