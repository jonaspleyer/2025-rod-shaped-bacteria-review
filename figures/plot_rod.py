import pyvista as pv
import numpy as np

points = [
    [0, 0, 0],
    [1, 0.7, 0],
    [2, 2.1, 0],
    [3, 2.6, 0],
    [4, 3.8, 0],
    [5, 4.7, 0],
]
points = np.array(points)


def points_in_cylinder(pt1, pt2, r, q):
    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    # cond1 = np.where(np.dot(q - pt1, vec)) >= 0 and np.dot(q - pt2, vec) <= 0
    # cond2 = np.linalg.norm(np.cross(q - pt1, vec)) <= const)
    # return cond1 and cond2


def remove_sphere_points(sphere, cylinder):
    selected = sphere.select_enclosed_points(cylinder)
    return sphere.remove_points(selected["SelectedPoints"].view(bool))[0]


if __name__ == "__main__":
    radius = 0.5

    # spheres = [ for p in points]

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

    import pyacvd

    clus = pyacvd.Clustering(mesh)
    clus.subdivide(2)
    clus.cluster(2000, iso_try=20)
    mesh = clus.create_mesh().compute_normals()

    # Prepare perlin noise
    freq1 = [0.689, 0.562, 0.683]
    freq2 = [16, 15, 14]
    noise1 = pv.perlin_noise(0.01, freq1, (0, 0, 0))
    noise2 = pv.perlin_noise(0.05, freq2, (0, 0, 0))

    mesh["scalars1"] = [noise1.EvaluateFunction(p) for p in mesh.points]
    mesh = mesh.warp_by_scalar("scalars1")
    mesh.smooth(n_iter=10, inplace=True)
    mesh.subdivide(3)
    mesh["scalars2"] = [noise2.EvaluateFunction(p) for p in mesh.points]
    mesh = mesh.warp_by_scalar("scalars2")

    mesh.plot(
        # show_edges=True,
        # opacity=0.5,
        show_scalar_bar=False,
        scalars=None,
        color=[150, 150, 150],
        smooth_shading=True,
        metallic=1,
        roughness=0,
    )
