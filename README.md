# pyquadrat
Python module and associated notebook for calculating quadrature points and weights

## usage:

```import pyquad

ndim = 3
npoly = 3

g = gauss_points_box(npoly,ndim)
gm = gauss_map_to_simplex(g)

g.tofile("gauss_box.txt")
gm.tofile("gauss_simplex.txt")
```
