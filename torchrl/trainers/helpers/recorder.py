# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser

__all__ = ["parser_recorder_args"]


def parser_recorder_args(parser: ArgumentParser) -> ArgumentParser:
    """
    Populates the argument parser to build a recorder.

    Args:
        parser (ArgumentParser): parser to be populated.

    """

    parser.add_argument(
        "--record_video",
        "--record-video",
        action="store_true",
        help="whether a video of the task should be rendered during logging.",
    )
    parser.add_argument(
        "--no_video",
        "--no-video",
        action="store_false",
        dest="record_video",
        help="whether a video of the task should be rendered during logging.",
    )
    parser.add_argument(
        "--exp_name",
        "--exp-name",
        type=str,
        default="",
        help="experiment name. Used for logging directory. "
        "A date and uuid will be joined to account for multiple experiments with the same name.",
    )
    parser.add_argument(
        "--record_interval",
        "--record-interval",
        type=int,
        default=1000,
        help="number of batch collections in between two collections of validation rollouts. "
        "Default=1000.",
    )
    parser.add_argument(
        "--record_frames",
        "--record-frames",
        type=int,
        default=1000,
        help="number of steps in validation rollouts. " "Default=1000.",
    )

    return parser
