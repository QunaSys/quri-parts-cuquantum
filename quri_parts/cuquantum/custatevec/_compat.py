# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility shim for the cuQuantum SDK 25.03+ namespace rename.

Pre-25.03: ``custatevec`` was a submodule of ``cuquantum``.
25.03+:    it moved under ``cuquantum.bindings``.

This module exposes a single ``custatevec`` name that resolves to whichever
location is available at runtime, so callers can simply do::

    from ._compat import cuquantum, custatevec
    custatevec.create()  # works on both old and new SDKs
"""
try:
    import cuquantum

    try:
        from cuquantum.bindings import custatevec  # SDK 25.03+
    except ImportError:
        from cuquantum import custatevec  # pre-25.03
except ImportError:
    cuquantum = None
    custatevec = None

__all__ = ["cuquantum", "custatevec"]
