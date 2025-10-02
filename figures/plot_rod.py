import pyvista as pv
import numpy as np
import pyacvd

points = [
    [0, 0, 0],
    [0.5, 0.3, 0],
    [1, 0.8, 0],
    [1.5, 1.5, 0],
    [2, 2.1, 0],
    [3, 2.99, 0],
    [4, 3.8, 0],
    [5, 4.7, 0],
]
points = np.array(points)


def remove_sphere_points(sphere, cylinder):
    selected = sphere.select_enclosed_points(cylinder)
    return sphere.remove_points(selected["SelectedPoints"].view(bool))[0]


def create_initial_mesh(seed=0):
    radius = 0.5

    theta_resolution = 30
    meshes = [
        pv.Sphere(
            center=points[0],
            radius=radius,
            direction=points[1] - points[0],
            theta_resolution=theta_resolution,
        ).subdivide(2)
    ]
    for p1, p2 in zip(points[:-1], points[1:]):
        center = 0.5 * (p1 + p2)
        direction = p2 - p1
        height = np.linalg.norm(direction)
        direction /= height

        sphere = pv.Sphere(
            center=p2,
            radius=radius,
            direction=p2 - p1,
            theta_resolution=theta_resolution,
        ).subdivide(2)
        cylinder_cmp = pv.Cylinder(
            center=center,
            radius=radius,
            height=height,
            direction=direction,
        ).clean()

        sphere = remove_sphere_points(sphere, cylinder_cmp)
        meshes[-1] = remove_sphere_points(meshes[-1], cylinder_cmp)
        meshes.append(
            pv.CylinderStructured(
                center=center,
                radius=radius,
                height=height,
                direction=direction,
                theta_resolution=theta_resolution,
            )
            .extract_surface()
            .triangulate()
            .subdivide(2)
        )
        meshes.append(sphere)

    # Final sphere
    meshes[-1] = remove_sphere_points(meshes[-1], cylinder_cmp)

    pcloud = np.vstack([p.points for p in meshes])
    cloud = pv.PolyData(pcloud)
    mesh = cloud.reconstruct_surface().clean()
    mesh.compute_normals(inplace=True)

    clus = pyacvd.Clustering(mesh)
    clus.subdivide(2)
    clus.cluster(2000, iso_try=20)
    mesh = clus.create_mesh().compute_normals()

    # Prepare perlin noise
    freq1 = [0.689, 0.562, 0.683]
    freq2 = [50, 50, 50]
    noise1 = pv.perlin_noise(0.05, freq1, (0, 0, 0))

    rng = np.random.default_rng(seed)

    def noise2(_p):
        return rng.normal(scale=0.02)

    mesh["scalars1"] = [noise1.EvaluateFunction(p) for p in mesh.points]
    # mesh.smooth(n_iter=250, inplace=True)
    mesh.subdivide(4, inplace=True)
    mesh["scalars2"] = [noise2(p) for p in mesh.points]
    mesh = mesh.warp_by_scalar("scalars2")
    mesh.smooth(n_iter=250, inplace=True)

    return mesh, radius, points


if __name__ == "__main__":
    mesh, radius, points = create_initial_mesh()
    rng = np.random.default_rng(0)
    mesh["colors"] = rng.normal(180, scale=10, size=(len(mesh.points), 3))

    domain_size = np.max(points) + radius
    plotter = pv.Plotter(
        off_screen=True,
        window_size=[int(np.ceil(domain_size)) * 300] * 2,
    )
    p1 = np.array([-radius] * 3)
    p2 = np.array([domain_size, domain_size, -radius])
    p3 = np.array([-radius, domain_size, -radius])
    rect = pv.Rectangle(np.array([p1, p2, p3]))
    plotter.add_mesh(rect)
    bounds = (-radius, domain_size, -radius, domain_size, -radius, radius)
    pv.Plotter.view_xy(plotter, bounds=bounds)
    pv.Plotter.enable_parallel_projection(plotter)
    plotter.camera.tight(padding=0)
    plotter.camera.position = (*plotter.camera.position[:2], 100 * domain_size)
    camera = plotter.camera.copy()
    plotter.clear_actors()

    actor = plotter.add_mesh(
        mesh,
        # show_edges=True,
        show_scalar_bar=False,
        # scalars=None,
        # color=[150, 150, 150],
        scalars="colors",
        cmap="Grays",
        smooth_shading=True,
        metallic=1,
        roughness=0,
    )
    actor.UseBoundsOff()

    plotter.disable_anti_aliasing()
    plotter.screenshot("figures/rod_render.png")
    plotter.close()
