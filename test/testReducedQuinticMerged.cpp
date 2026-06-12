/**
 * @file testReducedQuinticMerged.cpp
 * @brief Comprehensive test suite for ReducedQuintic triangle elements
 * 
 * This file merges three separate test files:
 * - testReducedQuintic.cpp: Low-level internal tests
 * - testOmegahReducedQuintic.cpp: Mid-level API tests  
 * - testReducedQuinticAPI.cpp: High-level OmegahMeshField API tests
 * 
 * Tests are organized in three sections:
 * SECTION 1: Low-level internals (LU solver, geometry, coordinates)
 * SECTION 2: Mid-level API (getReducedQuinticTriangleElement)
 * SECTION 3: High-level API (OmegahMeshField::triangleReducedQuinticEval)
 */

#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Fail.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_For.hpp"
#include "MeshField_Macros.hpp"
#include "MeshField_ReducedQuintic.hpp"
#include "MeshField_Shape.hpp"
#include "MeshField_ShapeField.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_library.hpp"
#include "Omega_h_simplex.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace MeshField;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

// =============================================================================
// SECTION 1: LOW-LEVEL INTERNAL TESTS (from testReducedQuintic.cpp)
// =============================================================================

/**
 * @brief Test 1.1: LU solver
 * Tests the custom LU decomposition with partial pivoting
 */
bool test_LU_solver() {
  std::cout << "\n=== Test 1.1: LU Solver ===\n";
  
  const int n = 3;
  Real A[9] = {2, 1, 0,
               0, 2, 1,
               0, 0, 2};
  Real b[3] = {3, 3, 2};
  Real expected[3] = {1, 1, 1};
  
  int info = solveLU_internal(n, 1, A, n, b, 1);
  
  if (info != 0) {
    std::cout << "  ❌ FAILED: LU solver returned error code " << info << "\n";
    return false;
  }
  
  Real tol = 1e-10;
  bool passed = true;
  std::cout << "  Solution:\n";
  for (int i = 0; i < n; i++) {
    Real error = std::abs(b[i] - expected[i]);
    std::cout << "    x[" << i << "] = " << b[i] 
              << " (expected: " << expected[i] 
              << ", error: " << error << ")\n";
    if (error > tol) {
      passed = false;
    }
  }
  
  std::cout << (passed ? "  ✓ PASSED\n" : "  ❌ FAILED\n");
  return passed;
}

/**
 * @brief Test 1.2: Geometric parameter computation
 */
bool test_geometry_computation() {
  std::cout << "\n=== Test 1.2: Geometric Parameter Computation ===\n";
  
  Real coords[3][2] = {{0, 0}, {4, 0}, {0, 3}};
  Real origin[2], a, b, c, sin_theta, cos_theta;
  int order[3];
  computeReducedQuinticGeometry(coords, origin, a, b, c, sin_theta, cos_theta, order);
  
  std::cout << "  Triangle vertices: (0,0), (4,0), (0,3)\n";
  std::cout << "  Computed: a=" << a << ", b=" << b << ", c=" << c << "\n";
  std::cout << "            sin_theta=" << sin_theta << ", cos_theta=" << cos_theta << "\n";
  
  Real tol = 1e-6;
  bool passed = true;
  
  Real longest_edge = 5.0;  // sqrt(16 + 9) = 5
  if (std::abs(a + b - longest_edge) > tol) {
    std::cout << "  ❌ a + b != longest edge\n";
    passed = false;
  }
  
  if (a < 0 || b < 0 || c < 0) {
    std::cout << "  ❌ Negative geometric parameters\n";
    passed = false;
  }
  
  if (std::abs(sin_theta*sin_theta + cos_theta*cos_theta - 1.0) > tol) {
    std::cout << "  ❌ sin²θ + cos²θ != 1\n";
    passed = false;
  }
  
  std::cout << (passed ? "  ✓ PASSED\n" : "  ❌ FAILED\n");
  return passed;
}

/**
 * @brief Test 1.3: Coordinate transformation
 */
KOKKOS_INLINE_FUNCTION
Vector3 localToBarycentric(Vector2 const& local, Real a, Real b, Real c) {
  const Real xi = local[0];
  const Real eta = local[1];
  const Real lambda2 = eta / c;
  const Real rhs = 1.0 - lambda2;
  const Real lambda1 = (xi + b * rhs) / (a + b);
  const Real lambda0 = rhs - lambda1;
  return {lambda0, lambda1, lambda2};
}

bool test_coordinate_transformation() {
  std::cout << "\n=== Test 1.3: Coordinate Transformation ===\n";
  
  const Real tol = 1e-10;
  bool passed = true;
  
  // Test simple triangle
  {
    std::cout << "  Simple triangle:\n";
    const Real a = 4.0, b = 0.0, c = 3.0;
    Vector3 v0 = {1.0, 0.0, 0.0};
    int order[3] = {0, 1, 2};
    
    auto p0 = ReducedQuinticHelpers::barycentricToLocal(v0, order, a, b, c);
    passed &= (std::abs(p0[0] - 0.0) < tol && std::abs(p0[1] - 0.0) < tol);
    std::cout << "    vertex0: xi=" << p0[0] << ", eta=" << p0[1] 
              << (passed ? " ✓" : " ❌") << "\n";
  }
  
  // Test round-trip
  {
    std::cout << "  Round-trip test:\n";
    const Real a = 1.8, b = 3.2, c = 2.4;
    Vector2 local = {-0.2, 1.0};
    int order[3] = {0, 1, 2};
    
    auto bary = localToBarycentric(local, a, b, c);
    auto local2 = ReducedQuinticHelpers::barycentricToLocal(bary, order, a, b, c);
    
    bool roundtrip_ok = (std::abs(local2[0] - local[0]) < tol && 
                         std::abs(local2[1] - local[1]) < tol);
    passed &= roundtrip_ok;
    std::cout << "    local -> bary -> local: " << (roundtrip_ok ? "✓" : "❌") << "\n";
  }
  
  std::cout << (passed ? "  ✓ PASSED\n" : "  ❌ FAILED\n");
  return passed;
}

/**
 * @brief Test 1.4-1.9: Field evaluation with analytic functions
 */
struct EvalPoint {
  Real coord[2];  // Physical coordinates
  Real expectedVal;
  Real expectedDx;
  Real expectedDy;
};

bool test_field_evaluation(const char* testName, Real coords[3][2], Real dofs[18],
                           EvalPoint evalPoints[], int numPoints, Omega_h::Library& lib) {
  std::cout << "\n=== " << testName << " ===\n";
  
  // Precompute coefficients
  std::vector<Real> triCoords(6);
  for (int i = 0; i < 3; i++) {
    triCoords[i*2 + 0] = coords[i][0];
    triCoords[i*2 + 1] = coords[i][1];
  }
  
  // Get geometric parameters
  Real origin[2], a, b, c, sin_theta, cos_theta;
  int order[3];
  computeReducedQuinticGeometry(coords, origin, a, b, c, sin_theta, cos_theta, order);
  
  // Rotate DOFs to local coordinate system
  Real rotatedDofs[18];
  for(int i = 0; i < 18; i++) rotatedDofs[i] = dofs[i];
  for(int i = 0; i < 3; i++) {
    rotateDof(rotatedDofs + i*6, sin_theta, cos_theta);
  }
  
  auto elemCoeffs = precomputeReducedQuinticCoefficients(1, triCoords.data());
  auto elemCoeffs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), elemCoeffs);
  
  Real tol = 1e-8;
  bool passed = true;
  
  for (int p = 0; p < numPoints; p++) {
    Real physCoord[2] = {evalPoints[p].coord[0], evalPoints[p].coord[1]};
    
    // Convert physical to barycentric
    Real x0 = coords[0][0], y0 = coords[0][1];
    Real x1 = coords[1][0], y1 = coords[1][1];
    Real x2 = coords[2][0], y2 = coords[2][1];
    Real x = physCoord[0], y = physCoord[1];
    
    Real detT = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2);
    Real lambda0 = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / detT;
    Real lambda1 = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / detT;
    Real lambda2 = 1.0 - lambda0 - lambda1;
    
    Kokkos::Array<Real, 3> bary = {lambda0, lambda1, lambda2};
    
    // Evaluate on device
    ReducedQuinticTriangleShape shape;
    Kokkos::View<Real*> shapeValues_d("shapeValues", 18);
    Kokkos::View<Real**> shapeGrads_d("shapeGrads", 18, 2);
    
    Kokkos::parallel_for("EvaluateField", 1, KOKKOS_LAMBDA(int) {
      auto coeffSlice = Kokkos::subview(elemCoeffs, 0, Kokkos::ALL());
      auto shapeValues_array = shape.getValues(bary, coeffSlice);
      auto shapeGrads_array = shape.getLocalGradients(bary, coeffSlice);
      for (int i = 0; i < 18; i++) {
        shapeValues_d(i) = shapeValues_array[i];
        shapeGrads_d(i, 0) = shapeGrads_array[i][0];
        shapeGrads_d(i, 1) = shapeGrads_array[i][1];
      }
    });
    
    auto shapeValues = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shapeValues_d);
    auto shapeGrads = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shapeGrads_d);
    
    // Interpolate field value and gradients
    Real f_val = 0.0;
    Real dfdxi0 = 0.0;
    Real dfdxi1 = 0.0;
    
    for (int ni = 0; ni < 18; ni++) {
      f_val += shapeValues(ni) * rotatedDofs[ni];
      dfdxi0 += shapeGrads(ni, 0) * rotatedDofs[ni];
      dfdxi1 += shapeGrads(ni, 1) * rotatedDofs[ni];
    }
    
    // Transform gradients to physical coordinates
    Real J[2][2] = {
      {coords[0][0] - coords[2][0], coords[1][0] - coords[2][0]},
      {coords[0][1] - coords[2][1], coords[1][1] - coords[2][1]}
    };
    Real detJ = J[0][0]*J[1][1] - J[0][1]*J[1][0];
    Real Jinv[2][2] = {
      { J[1][1]/detJ, -J[0][1]/detJ},
      {-J[1][0]/detJ,  J[0][0]/detJ}
    };
    
    Real dfdx = Jinv[0][0]*dfdxi0 + Jinv[1][0]*dfdxi1;
    Real dfdy = Jinv[0][1]*dfdxi0 + Jinv[1][1]*dfdxi1;
    
    // Compare with expected values
    Real errVal = std::abs(f_val - evalPoints[p].expectedVal);
    Real errDx = std::abs(dfdx - evalPoints[p].expectedDx);
    Real errDy = std::abs(dfdy - evalPoints[p].expectedDy);
    
    bool ptPassed = (errVal < tol && errDx < tol && errDy < tol);
    passed &= ptPassed;
    
    if (!ptPassed) {
      std::cout << "  Point " << p << " (" << physCoord[0] << "," << physCoord[1] << "): "
                << "val=" << f_val << " (exp " << evalPoints[p].expectedVal << ") "
                << (ptPassed ? "✓" : "❌") << "\n";
    }
  }
  
  std::cout << (passed ? "  ✓ PASSED\n" : "  ❌ FAILED\n");
  return passed;
}

bool run_low_level_field_tests(Omega_h::Library& lib) {
  bool allPassed = true;
  
  // Test 1.4: Constant field f=1
  {
    Real coords[3][2] = {{0, 0}, {4, 0}, {0, 3}};
    Real dofs[18];
    for (int i = 0; i < 3; i++) {
      dofs[i*6 + 0] = 1.0;
      for (int j = 1; j < 6; j++) dofs[i*6 + j] = 0.0;
    }
    EvalPoint evalPoints[] = {
      {{4.0/3.0, 1.0}, 1.0, 0.0, 0.0},  // centroid
      {{0.0, 0.0}, 1.0, 0.0, 0.0},      // vertex 0
      {{4.0, 0.0}, 1.0, 0.0, 0.0}       // vertex 1
    };
    allPassed &= test_field_evaluation("Test 1.4: Constant field (f=1)", 
                                       coords, dofs, evalPoints, 3, lib);
  }
  
  // Test 1.5: Linear field f=x
  {
    Real coords[3][2] = {{0, 0}, {4, 0}, {0, 3}};
    Real dofs[18];
    for (int i = 0; i < 3; i++) {
      dofs[i*6 + 0] = coords[i][0];  // f = x
      dofs[i*6 + 1] = 1.0;            // df/dx = 1
      dofs[i*6 + 2] = 0.0;            // df/dy = 0
      for (int j = 3; j < 6; j++) dofs[i*6 + j] = 0.0;
    }
    EvalPoint evalPoints[] = {
      {{4.0/3.0, 1.0}, 4.0/3.0, 1.0, 0.0},  // centroid
      {{2.0, 0.0}, 2.0, 1.0, 0.0}            // midpoint on edge
    };
    allPassed &= test_field_evaluation("Test 1.5: Linear field (f=x)", 
                                       coords, dofs, evalPoints, 2, lib);
  }
  
  // Test 1.6: Quadratic field f=x²
  {
    Real coords[3][2] = {{0, 0}, {4, 0}, {0, 3}};
    Real dofs[18];
    for (int i = 0; i < 3; i++) {
      Real x = coords[i][0];
      dofs[i*6 + 0] = x * x;
      dofs[i*6 + 1] = 2.0 * x;
      dofs[i*6 + 2] = 0.0;
      dofs[i*6 + 3] = 2.0;
      dofs[i*6 + 4] = 0.0;
      dofs[i*6 + 5] = 0.0;
    }
    EvalPoint evalPoints[] = {
      {{4.0/3.0, 1.0}, (4.0/3.0)*(4.0/3.0), 2.0*(4.0/3.0), 0.0},  // centroid
      {{2.0, 0.0}, 4.0, 4.0, 0.0}                                   // edge midpoint
    };
    allPassed &= test_field_evaluation("Test 1.6: Quadratic field (f=x²)", 
                                       coords, dofs, evalPoints, 2, lib);
  }
  
  return allPassed;
}

// =============================================================================
// SECTION 2: MID-LEVEL API TEST (from testOmegahReducedQuintic.cpp)
// =============================================================================

/**
 * @brief Test 2: Mid-level API using getReducedQuinticTriangleElement
 */
bool test_mid_level_api() {
  std::cout << "\n=== Test 2: Mid-Level API (getReducedQuinticTriangleElement) ===\n";
  
  Omega_h::Library lib;
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto mesh = Omega_h::build_box(world, family, 1.0, 1.0, 0.0, 1, 1, 0);
  
  auto coords = mesh.coords();
  auto coords_h = Omega_h::HostRead<Omega_h::Real>(coords);
  
  // Get ReducedQuintic element via high-level API
  auto [shape, map, elemCoeffs] = Omegah::getReducedQuinticTriangleElement(mesh);
  
  std::cout << "  ReducedQuintic element created, " << mesh.nfaces() << " triangles\n";
  
  // Create field with 6 components per vertex
  const size_t numComponents = 6;
  using Ctrlr = KokkosController<MemorySpace, ExecutionSpace, Real***>;
  Ctrlr controller({mesh.nverts(), 1, numComponents});
  
  auto vtxField = makeField<Ctrlr, 0>(controller);
  using LA = LinearAccessor<decltype(vtxField)>;
  using FieldType = ShapeField<numComponents, Ctrlr, LinearTriangleShape, LA>;
  
  MeshInfo meshInfo{mesh.nverts(), mesh.nedges(), mesh.nfaces(), 0};
  FieldType field(controller, meshInfo, {vtxField});
  
  // Set DOFs for linear function f(x,y) = 2x + y
  Kokkos::parallel_for(
      "setDOFs", mesh.nverts(), KOKKOS_LAMBDA(const int &v) {
        Real x = coords[v*2];
        Real y = coords[v*2 + 1];
        field(v, 0, 0, Vertex) = 2*x + y;
        field(v, 0, 1, Vertex) = 2.0;
        field(v, 0, 2, Vertex) = 1.0;
        field(v, 0, 3, Vertex) = 0.0;
        field(v, 0, 4, Vertex) = 0.0;
        field(v, 0, 5, Vertex) = 0.0;
      });
  
  // Create FieldElement
  MeshField::FieldElement<FieldType, decltype(shape), decltype(map)> fieldElem(
      mesh.nfaces(), field, shape, map, elemCoeffs);
  
  // Test points: centroid and vertices
  auto numFaces = mesh.nfaces();
  Kokkos::View<Real*[3]> testPoints("testPoints", numFaces * 4);
  auto testPoints_h = Kokkos::create_mirror_view(testPoints);
  
  for (int e = 0; e < numFaces; ++e) {
    const int base = e * 4;
    // Centroid
    testPoints_h(base + 0, 0) = 1.0/3.0;
    testPoints_h(base + 0, 1) = 1.0/3.0;
    testPoints_h(base + 0, 2) = 1.0/3.0;
    // Vertices
    testPoints_h(base + 1, 0) = 1.0; testPoints_h(base + 1, 1) = 0.0; testPoints_h(base + 1, 2) = 0.0;
    testPoints_h(base + 2, 0) = 0.0; testPoints_h(base + 2, 1) = 1.0; testPoints_h(base + 2, 2) = 0.0;
    testPoints_h(base + 3, 0) = 0.0; testPoints_h(base + 3, 1) = 0.0; testPoints_h(base + 3, 2) = 1.0;
  }
  Kokkos::deep_copy(testPoints, testPoints_h);
  
  auto result = MeshField::evaluate(fieldElem, testPoints, 4);
  auto result_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  
  // Verify results
  auto triVerts_d = mesh.ask_elem_verts();
  Omega_h::HostRead triVerts(triVerts_d);
  
  bool allPassed = true;
  Real tol = 1e-8;
  const char* pointNames[] = {"centroid", "vertex 0", "vertex 1", "vertex 2"};
  
  for (int p = 0; p < 4; p++) {
    Real x = 0.0, y = 0.0;
    for (int vi = 0; vi < 3; vi++) {
      const auto triDim = 2;
      const auto vtxDim = 0;
      const auto ignored = -1;
      const auto localVtxIdx = (Omega_h::simplex_down_template(triDim, vtxDim, vi, ignored) + 2) % 3;
      const auto triToVtxDegree = Omega_h::simplex_degree(triDim, vtxDim);
      const Omega_h::LO vtxIdx = triVerts[0 * triToVtxDegree + localVtxIdx];
      
      x += testPoints_h(p, vi) * coords_h[vtxIdx*2];
      y += testPoints_h(p, vi) * coords_h[vtxIdx*2 + 1];
    }
    
    Real expected = 2*x + y;
    Real computed = result_h(p, 0);
    Real error = std::abs(computed - expected);
    bool passed = error < tol;
    allPassed &= passed;
    
    std::cout << "  " << pointNames[p] << ": computed=" << computed 
              << ", expected=" << expected << " " << (passed ? "✓" : "❌") << "\n";
  }
  
  std::cout << (allPassed ? "  ✓ PASSED\n" : "  ❌ FAILED\n");
  return allPassed;
}

// =============================================================================
// SECTION 3: HIGH-LEVEL API TEST (from testReducedQuinticAPI.cpp)
// =============================================================================

struct LinearFunction {
  KOKKOS_INLINE_FUNCTION
  Real operator()(Real x, Real y) const { return 2.0 * x + y; }
  static constexpr Real dfdx = 2.0;
  static constexpr Real dfdy = 1.0;
  static constexpr Real d2fdx2 = 0.0;
  static constexpr Real d2fdxy = 0.0;
  static constexpr Real d2fdy2 = 0.0;
};

struct QuadraticFunction {
  KOKKOS_INLINE_FUNCTION
  Real operator()(Real x, Real y) const { return (x * x) + (2.0 * y); }
};

struct TestCoords {
  Kokkos::View<Real *[3]> coords;
  size_t NumPtsPerElem;
  std::string name;
};

template <size_t NumPtsPerElem>
Kokkos::View<Real *[3]>
createElmAreaCoords(size_t numElements,
                    Kokkos::Array<Real, 3 * NumPtsPerElem> coords) {
  Kokkos::View<Real *[3]> lc("localCoords", numElements * NumPtsPerElem);
  Kokkos::parallel_for(
      "setLocalCoords", numElements, KOKKOS_LAMBDA(const int &elm) {
        for (size_t pt = 0; pt < NumPtsPerElem; pt++) {
          lc(elm * NumPtsPerElem + pt, 0) = coords[pt * 3 + 0];
          lc(elm * NumPtsPerElem + pt, 1) = coords[pt * 3 + 1];
          lc(elm * NumPtsPerElem + pt, 2) = coords[pt * 3 + 2];
        }
      });
  return lc;
}

template <typename Result, typename CoordField, typename AnalyticFunction>
bool checkResult(Omega_h::Mesh &mesh, Result &result, CoordField coordField,
                 TestCoords testCase, AnalyticFunction func, size_t numComp) {
  const size_t npts_per_elem = testCase.NumPtsPerElem;
  const size_t num_comp = numComp;
  
  MeshField::FieldElement fcoords(
      mesh.nfaces(), coordField, MeshField::LinearTriangleCoordinateShape(),
      MeshField::Omegah::LinearTriangleToVertexField(mesh));
  auto globalCoords = MeshField::evaluate(fcoords, testCase.coords, npts_per_elem);

  // Print summary of test dimensions
  std::cout << "  [checkResult] test=\"" << testCase.name
            << "\", nfaces=" << mesh.nfaces()
            << ", npts_per_elem=" << npts_per_elem
            << ", numComp=" << num_comp
            << ", result.extent(0)=" << result.extent(0)
            << ", globalCoords.extent(0)=" << globalCoords.extent(0)
            << ", expected total pts=" << (mesh.nfaces() * npts_per_elem) << "\n";

  // Print first few element coordinates and results (host-side copy)
  {
    auto globalCoords_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), globalCoords);
    auto result_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
    
    size_t printLimit = std::min<size_t>(mesh.nfaces() * npts_per_elem, 9);
    for (size_t pt = 0; pt < printLimit; pt++) {
      Real x = globalCoords_h(pt, 0);
      Real y = globalCoords_h(pt, 1);
      Real expected = func(x, y);
      Real computed = result_h(pt, 0);
      Real diff = std::abs(computed - expected);
      std::cout << "    pt=" << pt << " (elm=" << (pt / npts_per_elem)
                << "): x=" << x << " y=" << y
                << " expected=" << expected
                << " computed=" << computed
                << " diff=" << diff
                << (diff > MeshField::MachinePrecision ? " ❌" : " ✓")
                << "\n";
    }
    if (mesh.nfaces() * npts_per_elem > printLimit) {
      std::cout << "    ... (showing first " << printLimit << " of "
                << (mesh.nfaces() * npts_per_elem) << " points)\n";
    }
  }

  LO numErrors = 0;
  Kokkos::parallel_reduce(
      "checkResult", mesh.nfaces(),
      KOKKOS_LAMBDA(const int &ent, LO &lerrors) {
        const auto first = ent * npts_per_elem;
        const auto last = first + npts_per_elem;
        
        for (auto pt = first; pt < last; pt++) {
          const auto x = globalCoords(pt, 0);
          const auto y = globalCoords(pt, 1);
          const auto expected = func(x, y);
          
          for (size_t i = 0; i < num_comp; ++i) {
            const auto computed = result(pt, i);
            const auto diff = Kokkos::fabs(computed - expected);
            LO isError = 0;
            if (diff > MeshField::MachinePrecision) {
              isError = 1;
              Kokkos::printf(
                  "    ERROR: ent=%d pt=%d comp=%zu: expected=%.6f computed=%.6f diff=%.15e (x=%.6f y=%.6f)\n",
                  ent, (int)pt, i, expected, computed, diff, x, y);
            }
            lerrors += isError;
          }
        }
      },
      numErrors);
  
  std::cout << "  [checkResult] numErrors=" << numErrors << "\n";
  
  // Determine function name for error message
  const char* funcName = "unknown";
  if constexpr (std::is_same_v<AnalyticFunction, LinearFunction>) {
    funcName = "linear";
  } else if constexpr (std::is_same_v<AnalyticFunction, QuadraticFunction>) {
    funcName = "quadratic";
  }
  
  if (numErrors > 0) {
    fail("checkResult: %d error(s) found for test \"%s\" (func=%s, numComp=%zu)\n",
         static_cast<int>(numErrors), testCase.name.c_str(), funcName, numComp);
  }
  
  return false;  // no errors means test passed
}

template <typename AnalyticFunction, typename ShapeField>
void setReducedQuinticDOFs(Omega_h::Mesh &mesh, AnalyticFunction func,
                           ShapeField field) {
  const auto MeshDim = mesh.dim();
  auto coords = mesh.coords();
  auto setFieldAtVertices = KOKKOS_LAMBDA(const int &vtx) {
    const auto x = coords[vtx * MeshDim];
    const auto y = coords[vtx * MeshDim + 1];
    field(vtx, 0, 0, Vertex) = func(x, y);
    if constexpr (std::is_same_v<AnalyticFunction, LinearFunction>) {
      field(vtx, 0, 1, Vertex) = LinearFunction::dfdx;
      field(vtx, 0, 2, Vertex) = LinearFunction::dfdy;
      field(vtx, 0, 3, Vertex) = LinearFunction::d2fdx2;
      field(vtx, 0, 4, Vertex) = LinearFunction::d2fdxy;
      field(vtx, 0, 5, Vertex) = LinearFunction::d2fdy2;
    } else {
      field(vtx, 0, 1, Vertex) = 2.0 * x;
      field(vtx, 0, 2, Vertex) = 2.0;
      field(vtx, 0, 3, Vertex) = 2.0;
      field(vtx, 0, 4, Vertex) = 0.0;
      field(vtx, 0, 5, Vertex) = 0.0;
    }
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {mesh.nverts()},
                          setFieldAtVertices, "setReducedQuinticDOFs");
}

template <size_t numComponents, template <typename...> typename Controller,
          typename TestCaseType, typename FunctionType>
bool runHighLevelTest(Omega_h::Mesh &mesh,
                      OmegahMeshField<ExecutionSpace, 2, Controller> &omf,
                      TestCaseType testCase, FunctionType function,
                      const char* testName) {
  std::cout << "\n  --- Running: " << testName << " (numComponents=" << numComponents << ") ---\n";
  
  using ViewType = decltype(testCase.coords);
  auto field = omf.template CreateLagrangeField<Real, 1, 6>();
  using FieldType = decltype(field);
  setReducedQuinticDOFs(mesh, function, field);
  
  auto result = omf.template triangleReducedQuinticEval<ViewType, FieldType>(
      testCase.coords, testCase.NumPtsPerElem, field);
  
  std::cout << "  [runHighLevelTest] result.extent(0)=" << result.extent(0)
            << ", result.extent(1)=" << result.extent(1) << "\n";
  
  auto failed = checkResult(mesh, result, omf.getCoordField(), testCase,
                            decltype(function){}, numComponents);
  
  std::cout << "    " << testName << ": " << (failed ? "❌ FAILED" : "✓ PASSED") << "\n";
  return !failed;
}

template <template <typename...> typename Controller>
bool run_high_level_tests(Omega_h::Mesh &mesh,
                          OmegahMeshField<ExecutionSpace, 2, Controller> &omf,
                          const char* controllerName) {
  std::cout << "  Testing with " << controllerName << ":\n";
  
  static const size_t OnePtPerElem = 1;
  auto centroids = createElmAreaCoords<OnePtPerElem>(
      mesh.nfaces(), {1/3.0, 1/3.0, 1/3.0});
  auto interior = createElmAreaCoords<OnePtPerElem>(
      mesh.nfaces(), {0.1, 0.4, 0.5});
  
  const auto cases = {
    TestCoords{centroids, OnePtPerElem, "centroids"},
    TestCoords{interior, OnePtPerElem, "interior"}
  };
  
  bool allPassed = true;
  for (auto testCase : cases) {
    allPassed &= runHighLevelTest<1>(mesh, omf, testCase, LinearFunction(), 
                                     (testCase.name + " linear").c_str());
    allPassed &= runHighLevelTest<1>(mesh, omf, testCase, QuadraticFunction(), 
                                     (testCase.name + " quadratic").c_str());
  }
  
  return allPassed;
}

bool test_high_level_api() {
  std::cout << "\n=== Test 3: High-Level API (OmegahMeshField::triangleReducedQuinticEval) ===\n";
  
  Omega_h::Library lib;
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto mesh = Omega_h::build_box(world, family, 1.0, 1.0, 0.0, 3, 3, 0);
  
  bool allPassed = true;
  
  // Test with KokkosController
  {
    OmegahMeshField<ExecutionSpace, 2, KokkosController> omf(mesh);
    allPassed &= run_high_level_tests(mesh, omf, "KokkosController");
  }
  
#ifdef MESHFIELDS_ENABLE_CABANA
  // Test with CabanaController
  {
    auto mesh2 = Omega_h::build_box(world, family, 1.0, 1.0, 0.0, 3, 3, 0);
    OmegahMeshField<ExecutionSpace, 2, CabanaController> omf(mesh2);
    allPassed &= run_high_level_tests(mesh2, omf, "CabanaController");
  }
#endif
  
  std::cout << (allPassed ? "  ✓ PASSED\n" : "  ❌ FAILED\n");
  return allPassed;
}

// =============================================================================
// MAIN: Run all test sections
// =============================================================================

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  
  std::cout << "\n";
  std::cout << "=================================================================\n";
  std::cout << "   REDUCED QUINTIC COMPREHENSIVE TEST SUITE (MERGED)           \n";
  std::cout << "=================================================================\n";
  
  bool allPassed = true;
  
  // SECTION 1: Low-level internal tests
  std::cout << "\n";
  std::cout << "╔═════════════════════════════════════════════════════════════╗\n";
  std::cout << "║  SECTION 1: LOW-LEVEL INTERNAL TESTS                       ║\n";
  std::cout << "╚═════════════════════════════════════════════════════════════╝\n";
  
  allPassed &= test_LU_solver();
  allPassed &= test_geometry_computation();
  allPassed &= test_coordinate_transformation();
  allPassed &= run_low_level_field_tests(lib);
  
  // SECTION 2: Mid-level API tests
  std::cout << "\n";
  std::cout << "╔═════════════════════════════════════════════════════════════╗\n";
  std::cout << "║  SECTION 2: MID-LEVEL API TESTS                            ║\n";
  std::cout << "╚═════════════════════════════════════════════════════════════╝\n";
  
  allPassed &= test_mid_level_api();
  
  // SECTION 3: High-level API tests
  std::cout << "\n";
  std::cout << "╔═════════════════════════════════════════════════════════════╗\n";
  std::cout << "║  SECTION 3: HIGH-LEVEL API TESTS                           ║\n";
  std::cout << "╚═════════════════════════════════════════════════════════════╝\n";
  
  allPassed &= test_high_level_api();
  
  // Final summary
  std::cout << "\n";
  std::cout << "=================================================================\n";
  if (allPassed) {
    std::cout << "   ✓✓✓ ALL TESTS PASSED ✓✓✓                                   \n";
  } else {
    std::cout << "   ❌❌❌ SOME TESTS FAILED ❌❌❌                                \n";
  }
  std::cout << "=================================================================\n";
  
  Kokkos::finalize();
  return allPassed ? 0 : 1;
}
