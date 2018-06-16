#include "linalg.h"

inline void mat2scal(mat2 a, double b, mat2 c){
  vec2scal(a[0], b, c[0]);
  vec2scal(a[1], b, c[1]);
}

inline void mat2symm(mat2 m, mat2 s){
  mat2 mT;
  mat2trp(m, mT);
  mat2add(m, mT, s);
  mat2scal(s, 0.5, s);
}

namespace PCA2D{
inline void computeConsistentNodeValuesByPCA(vec2 *eigenvectors, int nodeCnt, double *PCAeigenvalues)
{ // compEV: Unstructured component for eigenvectors
  // length of eigenvectors is normalized for uniform weights
  // eigenValuesDesc: if not NULL, outputs eigenvalues in descending order
  // returns values in AVS node order
  /* scheme for achieving constistent orientation of the eigenvectors:
     1. do a PCA of all eigenvectors that belong to the cell
     2. choose an orientation for the principal eigenvector
     3. orient all eigenvectors according to the oriented principal eigenvector

     implementation: according to:
     http://en.wikipedia.org/wiki/Principal_components_analysis
  */

  // assemble data matrix X (each observation is a column vector)
  // each eigenvector produces 2 data points, one in each direction from origin
  float X[2][2*nodeCnt];
  {
    // go over nodes
    for (int n=0; n<nodeCnt; n++) {

      vec2 ev;
      vec2set(ev,eigenvectors[n][0],eigenvectors[n][1]);
      vec2nrm(ev, ev);

      X[0][n*2 + 0] = ev[0];
      X[1][n*2 + 0] = ev[1];

      X[0][n*2 + 1] = -ev[0];
      X[1][n*2 + 1] = -ev[1];
    }
  }

  // compute covariance matrix C
  // mean is already at origin because adding each eigenvector as two points
  // in both directions of the eigenvector, centered around origin and of
  // same length
  mat2 C;
  {
    for (int j=0; j<2; j++) {
      for (int i=0; i<2; i++) {

        double sum = 0.0;
        for (int k=0; k<2*nodeCnt; k++) {
          sum += X[i][k] * X[j][k];
        }

        C[i][j] = sum / (2*nodeCnt - 1);
      }
    }
  }

  // compute eigenvalues and eigenvectors
  vec2 eigenvalues;
  vec2 _eigenvectors[2];
  {
    // force C to be symmetric (added 2018-05-04, still untested)
    mat2symm(C, C);

    // eigenvalues
    bool allReal = (mat2eigenvalues(C, eigenvalues) == 2);

    if (!allReal) {
      //printf("got complex eigenvalues: %g, %g, %g, returning zero\n", eigenvalues[0], eigenvalues[1], eigenvalues[2]);
      //mat3dump(C, stdout);

      return;
    }

    // eigenvectors
    mat2realEigenvector(C, eigenvalues[0], _eigenvectors[0]);
    mat2realEigenvector(C, eigenvalues[1], _eigenvectors[1]);
    //mat3realEigenvector(C, eigenvalues[2], _eigenvectors[2]);
  }

#if 0
  // get largest eigenvalue
  int maxEVIdx;
  {
    if (eigenvalues[0] > eigenvalues[1]) {
      if (eigenvalues[0] > eigenvalues[2]) {
        maxEVIdx = 0;
      }
      else { // ev2 >= ev0 and ev0 > ev1
        maxEVIdx = 2;
      }
    }
    else { // ev1 >= ev0
      if (eigenvalues[2] > eigenvalues[1]) {
        maxEVIdx = 2;
      }
      else { // ev1 >= ev2 and ev1 >= ev0
        maxEVIdx = 1;
      }
    }
  }
#else
  // sort eigenvalues in descending order
  int evalDescIndices[2];
  {
    if(eigenvalues[0] > eigenvalues[1]){
      evalDescIndices[0] = 0;
      evalDescIndices[1] = 1;
    } else {
      evalDescIndices[0] = 1;
      evalDescIndices[1] = 0;
    }
  }
#endif

  //copy eigenvalues into outer array in descending order
  PCAeigenvalues[0] = eigenvalues[evalDescIndices[0]];
  PCAeigenvalues[1] = eigenvalues[evalDescIndices[1]];
  
  // get eigenvector belonging to largest eigenvalue
  // keep the sign (this is now the signed direction for the node set)
  vec2 evMax;
  {
    vec2copy(_eigenvectors[evalDescIndices[0]], evMax);
  }

  // orient eigenvectors at nodes
  {
    for (int n=0; n<nodeCnt; n++) {
      vec2 ev = { X[0][n*2+0], X[1][n*2+0] };

      if (vec2dot(ev, evMax) < 0) {
        // invert sign
        vec2scal(ev, -1.0, eigenvectors[n]);
      }
      else {
        // no inversion
        vec2copy(ev, eigenvectors[n]);
      }
    }
  }
}
}