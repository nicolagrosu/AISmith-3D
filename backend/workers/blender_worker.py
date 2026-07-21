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
from mathutils import Vector


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


def export_obj(input_path: str, output_path: str) -> None:
    """Export source geometry unchanged at the GLB/OBJ boundary."""
    clear_scene()
    import_model(input_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_materials=False,
        export_normals=True,
        export_uv=True,
        export_triangulated_mesh=True,
    )


def _join_meshes(meshes):
    meshes = [obj for obj in meshes if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh was imported")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def _ensure_uv(obj) -> None:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.015)
    bpy.ops.object.mode_set(mode="OBJECT")


def _mesh_ray_distance(meshes) -> float:
    points = [obj.matrix_world @ Vector(corner) for obj in meshes if obj.type == "MESH" for corner in obj.bound_box]
    if not points:
        return 0.01
    lo = [min(point[i] for point in points) for i in range(3)]
    hi = [max(point[i] for point in points) for i in range(3)]
    diagonal = math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))
    return max(0.001, diagonal * 0.025)


def _select_for_bake(source_meshes, target) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in source_meshes:
        obj.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target


def _bake_emit_channel(source_meshes, target, image, input_name: str, distance: float) -> None:
    """Bake a Principled input through emission without changing source files."""
    saved = []
    for obj in source_meshes:
        for material in obj.data.materials:
            if material is None:
                continue
            material.use_nodes = True
            nodes, links = material.node_tree.nodes, material.node_tree.links
            output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
            principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
            source_input = principled.inputs.get(input_name) if principled else None
            if output is None or source_input is None:
                continue
            original = [(link.from_socket, link.to_socket) for link in list(output.inputs["Surface"].links)]
            for link in list(output.inputs["Surface"].links):
                links.remove(link)
            emission = nodes.new("ShaderNodeEmission")
            emission.name = "AISmith temporary bake emission"
            if source_input.is_linked:
                links.new(source_input.links[0].from_socket, emission.inputs["Color"])
            else:
                value = source_input.default_value
                if isinstance(value, float):
                    emission.inputs["Color"].default_value = (value, value, value, 1.0)
                else:
                    emission.inputs["Color"].default_value = value
            links.new(emission.outputs[0], output.inputs["Surface"])
            saved.append((nodes, links, output, original, emission))
    try:
        _select_for_bake(source_meshes, target)
        bpy.ops.object.bake(type="EMIT", use_selected_to_active=True, cage_extrusion=distance, max_ray_distance=distance, margin=12)
    finally:
        for nodes, links, output, original, emission in saved:
            for link in list(output.inputs["Surface"].links):
                links.remove(link)
            for from_socket, to_socket in original:
                links.new(from_socket, to_socket)
            nodes.remove(emission)


def _activate_image(material, image):
    material.use_nodes = True
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    for item in material.node_tree.nodes:
        item.select = False
    node.select = True
    material.node_tree.nodes.active = node
    return node


def retopo_bake(source_path: str, target_path: str, output_path: str) -> None:
    """Bake source PBR appearance to a newly unwrapped Instant Meshes mesh."""
    clear_scene()
    import_model(source_path)
    source_meshes = list(selected_meshes())
    if not source_meshes:
        raise RuntimeError("The source GLB has no mesh to bake")
    import_model(target_path)
    target_meshes = [obj for obj in selected_meshes() if obj not in source_meshes]
    target = _join_meshes(target_meshes)
    target.name = "AISmith_Retopologized"
    _ensure_uv(target)

    target.data.materials.clear()
    material = bpy.data.materials.new("AISmith Retopologized Material")
    material.use_nodes = True
    target.data.materials.append(material)
    size = 2048
    base = bpy.data.images.new("AISmith_Retopo_BaseColor", width=size, height=size, alpha=True)
    roughness = bpy.data.images.new("AISmith_Retopo_Roughness", width=size, height=size, alpha=False)
    metallic = bpy.data.images.new("AISmith_Retopo_Metallic", width=size, height=size, alpha=False)
    for image in (roughness, metallic):
        image.colorspace_settings.name = "Non-Color"
    distance = _mesh_ray_distance(source_meshes)
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.render.bake.margin = 12

    for image, channel in ((base, "Base Color"), (roughness, "Roughness"), (metallic, "Metallic")):
        _activate_image(material, image)
        log(f"Baking {channel} texture")
        _bake_emit_channel(source_meshes, target, image, channel, distance)
        image.pack()

    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    textures = [node for node in nodes if node.type == "TEX_IMAGE"]
    base_node, rough_node, metallic_node = textures[-3:]
    links.new(base_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(metallic_node.outputs["Color"], bsdf.inputs["Metallic"])
    for obj in source_meshes:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
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
    elif op == "export-obj":
        export_obj(spec["input"], spec["output"])
    elif op == "retopo-bake":
        retopo_bake(spec["source"], spec["target"], spec["output"])
    elif op == "paint":
        paint(spec["input"], spec["output"], spec.get("settings", {}))
    elif op == "animate":
        animate(spec["input"], spec["output"], spec["library"], spec.get("animation", "Idle"))
    else:
        raise RuntimeError(f"Unknown Blender worker operation: {op}")
    log("Blender export complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
