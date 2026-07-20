"""Small, headless Blender boundary used by the local studio.

This file deliberately contains no UI code.  It is invoked as a short lived
process so Blender's allocator and image cache disappear between pipeline
stages.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import bpy


def log(message: str) -> None:
    print(json.dumps({"event": "log", "message": message}), flush=True)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def import_model(path: str) -> None:
    suffix = Path(path).suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=True)
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise RuntimeError(f"Blender cannot import {suffix} at this boundary")


def selected_meshes():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def export_glb(path: str, *, animations: bool = True) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = next(iter(selected_meshes()), None)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_animations=animations,
        export_skins=True,
        export_yup=True,
    )


def convert(input_path: str, output_path: str) -> None:
    clear_scene()
    import_model(input_path)
    export_glb(output_path, animations=False)


def hex_rgba(value: str) -> tuple[float, float, float, float]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError("paint color must be a six digit hex value")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,)


def paint(input_path: str, output_path: str, settings: dict) -> None:
    clear_scene()
    import_model(input_path)
    meshes = selected_meshes()
    if not meshes:
        raise RuntimeError("Texture paint requires at least one mesh")

    color = hex_rgba(str(settings.get("color", "#7c5cff")))
    roughness = max(0.0, min(1.0, float(settings.get("roughness", 42)) / 100.0))
    metallic = max(0.0, min(1.0, float(settings.get("metallic", 18)) / 100.0))
    size = max(256, min(2048, int(settings.get("texture_size", 1024))))
    image = bpy.data.images.new("AISmith_Painted_BaseColor", width=size, height=size, alpha=True)
    image.generated_color = color

    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        if obj.data.uv_layers.active is None:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(island_margin=0.02)
            bpy.ops.object.mode_set(mode="OBJECT")
        material = obj.active_material or bpy.data.materials.new("AISmith Material")
        obj.active_material = material
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        tex = nodes.get("AISmith Painted Texture") or nodes.new("ShaderNodeTexImage")
        tex.name = "AISmith Painted Texture"
        tex.image = image
        if not any(link.to_node == bsdf and link.to_socket == bsdf.inputs.get("Base Color") for link in links):
            links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic

    # Apply strokes in UV space.  This is intentionally deterministic and
    # works for both imported GLBs and the generated mesh.
    pixels = [list(color) for _ in range(size * size)]
    strokes = settings.get("strokes", [])
    if not strokes:
        strokes = [{"x": 0.5, "y": 0.5, "radius": 0.18, "color": settings.get("color", "#7c5cff"), "opacity": 1.0}]
    for stroke in strokes:
        sx = max(0.0, min(1.0, float(stroke.get("x", 0.5))))
        sy = max(0.0, min(1.0, float(stroke.get("y", 0.5))))
        radius = max(0.002, min(0.5, float(stroke.get("radius", 0.08))))
        rgba = hex_rgba(str(stroke.get("color", settings.get("color", "#7c5cff"))))
        alpha = max(0.0, min(1.0, float(stroke.get("opacity", 1.0))))
        min_x, max_x = int((sx - radius) * size), int((sx + radius) * size)
        min_y, max_y = int((sy - radius) * size), int((sy + radius) * size)
        for y in range(max(0, min_y), min(size, max_y + 1)):
            for x in range(max(0, min_x), min(size, max_x + 1)):
                distance = math.hypot(x / size - sx, y / size - sy)
                if distance > radius:
                    continue
                hardness = max(0.05, min(1.0, float(stroke.get("hardness", 0.8))))
                falloff = 1.0 if distance <= radius * hardness else (1.0 - distance / radius) / (1.0 - hardness)
                a = alpha * max(0.0, min(1.0, falloff))
                idx = y * size + x
                old = pixels[idx]
                pixels[idx] = [old[c] * (1 - a) + rgba[c] * a for c in range(4)]
    image.pixels = [channel for px in pixels for channel in px]
    image.pack()
    export_glb(output_path)


def action_curves(action):
    if getattr(action, "is_action_legacy", False):
        return action.fcurves
    curves = []
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                curves.extend(bag.fcurves)
    return curves


ALIASES = {
    "root": "root", "hips": "pelvis", "mixamorighips": "pelvis", "pelvis": "pelvis",
    "spine": "spine_01", "spine1": "spine_01", "spine01": "spine_01", "spine2": "spine_02", "spine3": "spine_03",
    "neck": "neck_01", "head": "head", "leftshoulder": "clavicle_l", "rightshoulder": "clavicle_r",
    "leftarm": "upperarm_l", "rightarm": "upperarm_r", "leftforearm": "lowerarm_l", "rightforearm": "lowerarm_r",
    "lefthand": "hand_l", "righthand": "hand_r", "leftupleg": "thigh_l", "rightupleg": "thigh_r",
    "leftleg": "calf_l", "rightleg": "calf_r", "leftfoot": "foot_l", "rightfoot": "foot_r",
}


def normalized(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def resolve_bone(name: str, target_names: set[str]) -> str | None:
    if name in target_names:
        return name
    clean = normalized(name)
    for target in target_names:
        if normalized(target) == clean:
            return target
    side = "_l" if any(x in clean for x in ("left", "_l", "lhand", "larm", "lleg")) else "_r" if any(x in clean for x in ("right", "_r", "rhand", "rarm", "rleg")) else ""
    base = clean.replace("left", "").replace("right", "").replace("mixamorig", "")
    mapped = ALIASES.get(clean) or ALIASES.get(base)
    if mapped and mapped in target_names:
        return mapped
    if mapped and side and f"{mapped.rstrip('_lr')}{side}" in target_names:
        return f"{mapped.rstrip('_lr')}{side}"
    return None


def animate(input_path: str, output_path: str, library_path: str, animation: str) -> None:
    clear_scene()
    import_model(input_path)
    target_armature = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if target_armature is None:
        raise RuntimeError("Animate requires a rigged GLB with an armature from the Auto Rig tab")
    target_names = {b.name for b in target_armature.data.bones}
    existing_objects = set(bpy.context.scene.objects)
    import_model(library_path)
    source_objects = [o for o in bpy.context.scene.objects if o not in existing_objects]
    source_armature = next((o for o in source_objects if o.type == "ARMATURE"), None)
    if source_armature is None:
        raise RuntimeError("Mesh2Motion library clip has no source armature")
    wanted = {"Idle": "Idle_A", "Walk": "Walk", "Run": "Sprint", "Wave": "Yes"}.get(animation, animation)
    source_action = bpy.data.actions.get(wanted) or next(iter(bpy.data.actions), None)
    if source_action is None:
        raise RuntimeError(f"Mesh2Motion clip '{wanted}' was not found")
    output_action = source_action.copy()
    output_action.name = f"AISmith_{animation}"
    matched = 0
    for curve in action_curves(output_action):
        if 'pose.bones["' not in curve.data_path:
            continue
        before = curve.data_path.split('pose.bones["', 1)[1].split('"', 1)[0]
        target = resolve_bone(before, target_names)
        if target is None:
            continue
        curve.data_path = curve.data_path.replace(f'pose.bones["{before}"]', f'pose.bones["{target}"]')
        matched += 1
    if matched < 8:
        raise RuntimeError(f"Animation '{animation}' is incompatible with this SkinToken skeleton ({matched} mapped channels)")
    target_armature.animation_data_create()
    target_armature.animation_data.action = output_action
    for obj in source_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for obj in bpy.context.scene.objects:
        obj.select_set(obj.type in {"MESH", "ARMATURE"})
    bpy.context.view_layer.objects.active = target_armature
    export_glb(output_path)
    log(f"Retargeted {matched} animation channels onto {len(target_names)} target bones")


def main() -> None:
    spec = json.loads(sys.argv[sys.argv.index("--") + 1])
    op = spec["op"]
    if op == "convert":
        convert(spec["input"], spec["output"])
    elif op == "paint":
        paint(spec["input"], spec["output"], spec.get("settings", {}))
    elif op == "animate":
        animate(spec["input"], spec["output"], spec["library"], spec.get("animation", "Idle"))
    else:
        raise RuntimeError(f"Unknown Blender worker operation: {op}")
    log("Blender export complete")


if __name__ == "__main__":
    main()
