#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_ReducedQuintic.hpp"
#include "MeshField_Shape.hpp"
#include "MeshField_ShapeField.hpp"
#include "KokkosController.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_library.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>

using namespace MeshField;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using Real = double;

/**
 * @brief Test using getReducedQuinticTriangleElement API
 * 
 * This test creates a mesh with one triangle, uses getReducedQuinticTriangleElement
 * to get the shape, map, and coefficients, creates a FieldElement, and evaluates it.
 */
/**
 * @brief Test using getReducedQuinticTriangleElement API
 * 
 * This test creates a mesh with one triangle, uses getReducedQuinticTriangleElement
 * to get the shape, map, and coefficients, creates a FieldElement, and evaluates it.
 */
bool testReducedQuinticElement() {
  std::cout << "Test: ReducedQuintic Element with getReducedQuinticTriangleElement API\n";
  std::cout << "=======================================================================\n\n";
  
  // Create a simple mesh with one triangle
  // We'll manually create a mesh with vertices at (0,0), (1,0), (0,1)
  Omega_h::Library lib;
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto mesh = Omega_h::build_box(world, family, 1.0, 1.0, 0.0, 1, 1, 0);
  
  // Verify mesh creation
  auto triVerts_d = mesh.ask_elem_verts();
  Omega_h::HostRead triVerts(triVerts_d);
  for (int e = 0; e < mesh.nfaces(); ++e) {
    std::cout << "Face " << e << ": ";
    for (int i = 0; i < 3; ++i)
      std::cout << triVerts[e*3+i] << " ";
    std::cout << "\n";
  }
  
  // Get mesh coordinates
  auto coords = mesh.coords();
  auto coords_h = Omega_h::HostRead<Omega_h::Real>(coords);
  std::cout << "Triangle vertices:\n";
  for (int i = 0; i < mesh.nverts(); i++) {
    std::cout << "  v" << i << " = (" << coords_h[i*2] << ", " << coords_h[i*2+1] << ")\n";
  }
  std::cout << "\n";
  
  // Use the high-level API to get ReducedQuintic element
  auto [shape, map, elemCoeffs] = Omegah::getReducedQuinticTriangleElement(mesh);
  
  std::cout << "ReducedQuintic element created with precomputed coefficients\n";
  std::cout << "  Number of triangles: " << mesh.nfaces() << "\n";
  std::cout << "  Coefficient view dimensions: " << elemCoeffs.extent(0) << " x " 
            << elemCoeffs.extent(1) << "\n\n";

  // debug print of coefficients
  auto elemCoeffs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), elemCoeffs);
  for(int i=0;i<6;i++)
  std::cout << "elemCoeffs[0][" << i << "] = "
            << elemCoeffs_h(0,i)
            << "\n";
  
  // Create a field with 6 components per vertex for ReducedQuintic DOFs
  // [value, ∂x, ∂y, ∂²x², ∂²xy, ∂²y²]
  const size_t numComponents = 6;
  using Ctrlr = KokkosController<MemorySpace, ExecutionSpace, Real***>;
  Ctrlr controller({mesh.nverts(), 1, numComponents});  // nverts, 1 node per vertex, 6 components
  
  auto vtxField = makeField<Ctrlr, 0>(controller);
  using LA = LinearAccessor<decltype(vtxField)>;
  using FieldType = ShapeField<numComponents, Ctrlr, LinearTriangleShape, LA>;
  
  MeshInfo meshInfo{mesh.nverts(), mesh.nedges(), mesh.nfaces(), 0};
  FieldType field(controller, meshInfo, {vtxField});
  
  std::cout << "Field created with " << numComponents << " components per vertex\n\n";
  
  // Set DOFs for linear function f(x,y) = 2x + y
  // ∂f/∂x = 2, ∂f/∂y = 1, all second derivatives = 0
  std::cout << "Setting DOFs for linear function f(x,y) = 2x + y:\n";
  Kokkos::parallel_for(
      "setDOFs", mesh.nverts(), KOKKOS_LAMBDA(const int &v) {
        Real x = coords[v*2];
        Real y = coords[v*2 + 1];
        Real f_val = 2*x + y;
        
        field(v, 0, 0, Vertex) = f_val;  // value
        field(v, 0, 1, Vertex) = 2.0;    // ∂f/∂x
        field(v, 0, 2, Vertex) = 1.0;    // ∂f/∂y
        field(v, 0, 3, Vertex) = 0.0;    // ∂²f/∂x²
        field(v, 0, 4, Vertex) = 0.0;    // ∂²f/∂xy
        field(v, 0, 5, Vertex) = 0.0;    // ∂²f/∂y²
      });
  
  std::cout << "DOFs set for linear function f(x,y) = 2x + y\n\n";
  std::cout << "\n";
  Kokkos::parallel_for(
      "printDOFs", mesh.nverts(), KOKKOS_LAMBDA(const int &v) {
        Kokkos::printf("Vertex %d DOFs: f=%.2f, df/dx=%.2f, df/dy=%.2f, dxx=%.2f, dxy=%.2f, dyy=%.2f\n",
               v,
               field(v, 0, 0, Vertex),
               field(v, 0, 1, Vertex),
               field(v, 0, 2, Vertex),
               field(v, 0, 3, Vertex),
               field(v, 0, 4, Vertex),
               field(v, 0, 5, Vertex));
      });

  // Debug: print shape function values at centroid

  fprintf(stderr, "Debug: Shape function values at centroid:\n");
  ReducedQuinticTriangleShape dbgShape;
  fprintf(stderr, "  Centroid (1/3, 1/3, 1/3):\n");

  Kokkos::View<Real*> N_d("shapeValues", 18);
  Kokkos::parallel_for("EvaluateShapeAtCentroid", 1, KOKKOS_LAMBDA(int) {
    Kokkos::Array<Real,3> centroid =
    {
      1.0/3.0,
      1.0/3.0,
      1.0/3.0
    };
    auto coeffSlice = Kokkos::subview(elemCoeffs, 0, Kokkos::ALL());
    Kokkos::printf("  Evaluating shape functions at centroid (1/3, 1/3, 1/3)\n");
    auto shapeValues_array = dbgShape.getValues(centroid, coeffSlice);
    for (int i = 0; i < 18; i++) {
      N_d(i) = shapeValues_array[i];
    }
  });

  auto N = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), N_d);
  Real sumN = 0.0;

  for (int i = 0; i < 18; ++i)
    sumN += N(i);

  std::cout << "Sum of shape functions at centroid = "
            << std::setprecision(16)
            << sumN
            << "\n\n";

  Real directValue = 0.0;

  for (int v = 0; v < mesh.nverts(); ++v)
  {
    Real x = coords_h[v*2];
    Real y = coords_h[v*2+1];

    Real dofs[6] =
    {
      2*x + y,
      2.0,
      1.0,
      0.0,
      0.0,
      0.0
    };

    for (int d = 0; d < 6; ++d)
      directValue += N(v*6+d) * dofs[d];
  }

  std::cout
    << "Direct shape interpolation at centroid = "
    << directValue
    << "\n";

  Real cx = 0.0;
  Real cy = 0.0;

  for (int i = 0; i < 3; ++i)
  {
    int vtx = triVerts[i];

    cx += coords_h[vtx*2]   / 3.0;
    cy += coords_h[vtx*2+1] / 3.0;
  }

  std::cout
    << "Expected centroid value = "
    << (2.0*cx + cy)
    << "\n\n";

  std::cout << "Shape values at centroid:\n";

  for (int i = 0; i < 18; ++i)
  {
    std::cout
      << "N[" << i << "] = "
      << std::setprecision(16)
      << N(i)
      << "\n";
  }

  std::cout << "\n";
  
  // Debug: Print the vertex mapping for triangle 0
  std::cout << "Debug: Vertex mapping for triangle 0:\n";
  for (int vi = 0; vi < 3; vi++) {
    const auto triDim = 2;
    const auto vtxDim = 0;
    const auto ignored = -1;
    const auto localVtxIdx = (Omega_h::simplex_down_template(triDim, vtxDim, vi, ignored) + 2) % 3;
    const auto triToVtxDegree = Omega_h::simplex_degree(triDim, vtxDim);
    const Omega_h::LO vtx = triVerts[0 * triToVtxDegree + localVtxIdx];
    std::cout << "  vi=" << vi << " (barycentric index) -> localVtxIdx=" << localVtxIdx 
              << " -> global vertex=" << vtx 
              << " at (" << coords_h[vtx*2] << ", " << coords_h[vtx*2+1] << ")\n";
  }
  std::cout << "\n";
    
  
  // Create FieldElement with the ReducedQuintic shape, map, and coefficients
  FieldElement<FieldType, decltype(shape), decltype(map)> fieldElem(
      mesh.nfaces(), field, shape, map, elemCoeffs);
  
  std::cout << "FieldElement created\n\n";
  
  auto numFaces = mesh.nfaces();

  Kokkos::View<Real*[3]> testPoints("testPoints", numFaces * 5);
  auto testPoints_h = Kokkos::create_mirror_view(testPoints);

  for (int e = 0; e < numFaces; ++e) {
      const int base = e * 5;

      // Centroid
      testPoints_h(base + 0, 0) = 1.0 / 3.0;
      testPoints_h(base + 0, 1) = 1.0 / 3.0;
      testPoints_h(base + 0, 2) = 1.0 / 3.0;

      // Vertex 0
      testPoints_h(base + 1, 0) = 1.0;
      testPoints_h(base + 1, 1) = 0.0;
      testPoints_h(base + 1, 2) = 0.0;

      // Vertex 1
      testPoints_h(base + 2, 0) = 0.0;
      testPoints_h(base + 2, 1) = 1.0;
      testPoints_h(base + 2, 2) = 0.0;

      // Vertex 2
      testPoints_h(base + 3, 0) = 0.0;
      testPoints_h(base + 3, 1) = 0.0;
      testPoints_h(base + 3, 2) = 1.0;

      // Edge midpoint (vertex 0 - vertex 1)
      testPoints_h(base + 4, 0) = 0.5;
      testPoints_h(base + 4, 1) = 0.5;
      testPoints_h(base + 4, 2) = 0.0;
  }

  Kokkos::deep_copy(testPoints, testPoints_h);
  
  // Evaluate the field at test points
  auto result = evaluate(fieldElem, testPoints, 5);
  auto result_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  
  std::cout << "Evaluating field at test points:\n";
  std::cout << "--------------------------------\n";
  std::cout << "\nComparison:\n";

  for (int p = 0; p < 5; ++p)
  {
    std::cout
      << "point "
      << p
      << " fieldElem="
      << result_h(p,0)
      << "\n";
  }
  const char* pointNames[] = {"centroid", "vertex 0", "vertex 1", "vertex 2", "edge 0-1 midpoint"};
  bool allPassed = true;
  Real tol = 1e-8;
  
  for (int p = 0; p < 5; p++) {
    // Convert barycentric to physical coordinates using the correct vertex mapping
    Real x = 0.0, y = 0.0;
    for (int vi = 0; vi < 3; vi++) {
      // Apply the same transformation used in getReducedQuinticTriangleElement and ReducedQuinticTriangleToField
      const auto triDim = 2;
      const auto vtxDim = 0;
      const auto ignored = -1;
      const auto localVtxIdx = (Omega_h::simplex_down_template(triDim, vtxDim, vi, ignored) + 2) % 3;
      const auto triToVtxDegree = Omega_h::simplex_degree(triDim, vtxDim);
      const Omega_h::LO vtxIdx = triVerts[0 * triToVtxDegree + localVtxIdx];  // triangle 0
      
      x += testPoints_h(p, vi) * coords_h[vtxIdx*2];
      y += testPoints_h(p, vi) * coords_h[vtxIdx*2 + 1];
    }
    
    Real expected = 2*x + y;
    Real computed = result_h(p, 0);  // First component is the value
    Real error = std::abs(computed - expected);
    bool passed = error < tol;
    allPassed &= passed;
    
    std::cout << "  " << pointNames[p] << " (" << x << ", " << y << "):\n";
    std::cout << "    Expected: " << expected << "\n";
    std::cout << "    Computed: " << computed << "\n";
    std::cout << "    Error:    " << error << " " << (passed ? "✓" : "✗") << "\n\n";
  }
  
  if (allPassed) {
    std::cout << "✓ All tests PASSED\n\n";
  } else {
    std::cout << "✗ Some tests FAILED\n\n";
  }
  
  return allPassed;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    auto lib = Omega_h::Library(&argc, &argv);
    
    std::cout << "========================================\n";
    std::cout << "ReducedQuintic Element - Pipeline Test\n";
    std::cout << "========================================\n\n";
    
    bool passed = testReducedQuinticElement();
    
    if (!passed) {
      std::cout << "TEST FAILED\n";
      Kokkos::finalize();
      return 1;
    }
    
    std::cout << "========================================\n";
    std::cout << "All tests passed!\n";
    std::cout << "========================================\n";
  }
  Kokkos::finalize();
  return 0;
}