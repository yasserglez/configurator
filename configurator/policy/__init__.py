#
#  Copyright 2015 Yasser Gonzalez Fernandez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Configuration dialogs based on policies.
"""

from .dp import DPDialogBuilder
from .rl import RLDialogBuilder


__all__ = ["DPDialogBuilder", "RLDialogBuilder"]
