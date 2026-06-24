import argparse
import ast
import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import voxweave
from voxweave.cli import build_parser


def command_contract(parser):
    result = []
    for action in parser._actions:
        value = {
            "action": type(action).__name__,
            "dest": action.dest,
            "options": action.option_strings,
            "required": action.required,
            "nargs": action.nargs,
            "default": action.default,
            "const": action.const,
            "type": getattr(action.type, "__name__", None),
        }
        if isinstance(action, argparse._SubParsersAction):
            value["commands"] = {
                name: command_contract(child) for name, child in sorted(action.choices.items())
            }
        else:
            value["choices"] = action.choices
        result.append(value)
    return result


def snapshot():
    modules = {}
    for item in pkgutil.walk_packages(voxweave.__path__, voxweave.__name__ + "."):
        module = importlib.import_module(item.name)
        entries = {}
        for name, value in sorted(vars(module).items()):
            if name.startswith("_") or getattr(value, "__module__", None) != module.__name__:
                continue
            if inspect.isfunction(value):
                entries[name] = {"kind": "function", "signature": str(inspect.signature(value))}
            elif inspect.isclass(value):
                members = {}
                for member_name, member in sorted(vars(value).items()):
                    if member_name.startswith("_") and member_name not in {
                        "__len__",
                        "__iter__",
                        "__next__",
                        "__enter__",
                        "__exit__",
                        "__getitem__",
                        "__setitem__",
                        "__call__",
                    }:
                        continue
                    if isinstance(member, property):
                        members[member_name] = {
                            "kind": "property",
                            "get": str(inspect.signature(member.fget)),
                            "set": str(inspect.signature(member.fset)) if member.fset else None,
                        }
                    elif isinstance(member, (staticmethod, classmethod)):
                        members[member_name] = {
                            "kind": type(member).__name__,
                            "signature": str(inspect.signature(member.__func__)),
                        }
                    elif inspect.isfunction(member):
                        members[member_name] = {
                            "kind": "method",
                            "signature": str(inspect.signature(member)),
                        }
                tree = ast.parse(inspect.getsource(value))
                attributes = set()
                for node in ast.walk(tree):
                    targets = (
                        node.targets
                        if isinstance(node, ast.Assign)
                        else [node.target]
                        if isinstance(node, (ast.AnnAssign, ast.AugAssign))
                        else []
                    )
                    for target in targets:
                        for part in ast.walk(target):
                            if (
                                isinstance(part, ast.Attribute)
                                and isinstance(part.value, ast.Name)
                                and part.value.id == "self"
                                and not part.attr.startswith("_")
                            ):
                                attributes.add(part.attr)
                entries[name] = {
                    "kind": "class",
                    "signature": str(inspect.signature(value)),
                    "members": members,
                    "public_attributes": sorted(attributes),
                }
        constants = {}
        for node in ast.parse(inspect.getsource(module)).body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    constants[target.id] = ast.unparse(node.value)
        modules[item.name] = {"definitions": entries, "constants": constants}
    return {
        "schema_version": 1,
        "exports": voxweave.__all__,
        "modules": modules,
        "commands": command_contract(build_parser()),
    }


def test_complete_public_api_and_cli_contract():
    expected = json.loads((Path(__file__).parent / "fixtures/public-api.json").read_text())
    assert snapshot() == expected, (
        "Review the complete API diff before deliberately updating public-api.json"
    )
