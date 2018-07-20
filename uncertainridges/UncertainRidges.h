#ifndef __UNCERTAINRIDGES__
#define __UNCERTAINRIDGES__

#include "vtkImageAlgorithm.h"
#include "vtkMultiTimeStepAlgorithm.h"
#include "vtkSmartPointer.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDoubleArray.h"
#include "vtkCellData.h"
#include "vtkMultiBlockDataSet.h"
#include "chrono"
#include "vector"
#include <set>

//Override Eigen standard allocation limit
#ifndef EIGEN_STACK_ALLOCATION_LIMIT
#define EIGEN_STACK_ALLOCATION_LIMIT 1000000
#endif
#include <Eigen/Eigen>
#include <Eigen/unsupported/MatrixFunctions>
#include <omp.h>
#include "math.h"
#include "linalg.h"
#include "pca_incl.h"
#include "pca_incl_2D.h"
#include "mc_table.h"

typedef Eigen::Matrix<double,3,1> Vector3d;
typedef Eigen::Matrix<double,3,3> Matrix3d;

typedef Eigen::Matrix<double,2,1> Vector2d;
typedef Eigen::Matrix<double,2,2> Matrix2d;

typedef std::chrono::high_resolution_clock nanoClock;

typedef Eigen::Matrix<double,80,1> Vector80d;
typedef Eigen::Matrix<double,80,80> Matrix80d;
typedef Eigen::Matrix<std::complex<double>,80,1> Vector80c;
typedef Eigen::Matrix<std::complex<double>,80,80> Matrix80c;

typedef Eigen::Matrix<double,24,1> Vector24d;
typedef Eigen::Matrix<double,24,24> Matrix24d;
typedef Eigen::Matrix<std::complex<double>,24,1> Vector24c;
typedef Eigen::Matrix<std::complex<double>,24,24> Matrix24c;



class UncertainRidges : public vtkMultiTimeStepAlgorithm {
public:
    static UncertainRidges *New();
    vtkTypeMacro(UncertainRidges, vtkMultiTimeStepAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent) {};

    vtkSetMacro(numSamples, int);
    vtkGetMacro(numSamples, int);

    vtkSetMacro(extremum, int);
    vtkGetMacro(extremum, int);

    vtkSetMacro(computeLines, bool);
    vtkGetMacro(computeLines, bool);

    vtkSetMacro(calcCertain, bool);
    vtkGetMacro(calcCertain, bool);

    vtkSetMacro(useNewMethod, bool);
    vtkGetMacro(useNewMethod, bool);

    vtkSetMacro(useRandomSeed, bool);
    vtkGetMacro(useRandomSeed, bool);

    vtkSetMacro(useCholesky, bool);
    vtkGetMacro(useCholesky, bool);

    vtkSetClampMacro(evThresh, double, 0.001, 1.0);
    vtkGetMacro(evThresh, double);

    vtkSetMacro(evMin, double);
    vtkGetMacro(evMin, double);

    vtkSetClampMacro(crossTol, double, 0.0, 3.0);
    vtkGetMacro(crossTol, double);

protected:
    UncertainRidges();
    ~UncertainRidges() {};

    // Make sure the pipeline knows what type we expect as input
    int FillInputPortInformation( int port, vtkInformation* info );
    int FillOutputPortInformation( int port, vtkInformation* info );
    int RequestInformation(vtkInformation *vtkNotUsed(request), vtkInformationVector **inputVector, vtkInformationVector *outputVector); //the function that makes this class work with the vtk pipeline
    // Generate output
    int RequestData(vtkInformation *vtkNotUsed(request), vtkInformationVector **inputVector, vtkInformationVector *outputVector);
    int RequestUpdateExtent(vtkInformation*,vtkInformationVector** inputVector,vtkInformationVector* outputVector);

private:
    vtkSmartPointer<vtkImageData> data;
    vtkDoubleArray *extrProbability;
    vtkDoubleArray *cellProb;
    vtkSmartPointer<vtkDoubleArray> gradVecs;
    vtkSmartPointer<vtkDoubleArray> eps1;
    vtkSmartPointer<vtkDoubleArray> eps2;
    vtkSmartPointer<vtkDoubleArray> eps3;
    vtkSmartPointer<vtkDoubleArray> lambda1;
    vtkSmartPointer<vtkDoubleArray> lambda2;
    vtkSmartPointer<vtkDoubleArray> lambda3;

    int numSamples;
    int extremum;
    double *bounds;
    double *spacing;
    int *gridResolution;
    int offsetY;
    int offsetZ;
    int currentField;
    int arrayLength;
    nanoClock::time_point beginning;
    char *extrName;
    bool computeLines;
    bool useRandomSeed;
    bool useCholesky;
    std::mt19937 gen;
    double evThresh;
    double evMin;
    double crossTol;
    double spaceMag;
    bool is2D;
    int test;
    int sampleTest;
    bool useNewMethod;
    bool calcCertain;

    bool isCloseToEdge(int index);
    Vector80d generateNormalDistributedVec();
    Vector24d generateNormalDistributedVec2D();
    void computeGradients(Vector80d sampleVector, vec3 *gradients, mat3 *hessians, vec3 *secGrads, bool calcSec=false);
    void computeGradients2D(Vector24d sampleVector, vec2 *gradients, mat2 *hessians, vec2 *secGrads);
    double computeParVectors(vec3 *gradients, mat3 *hessians, vec3 *secGrads);
    int computeParallelOnCellface(vec3 *faceVel, vec3 *faceAcc, double *s, double *t);
    bool isRidgeOrValley(mat3 *hessians, vec3 *faceVel, double s, double t);
    bool computeRidgeLine2D(vec2 *gradients, mat2 *hessians, vec2 *secGrads);
    double computeRidgeLine2DTest(vec2 *gradients, mat2 *hessians, vec2 *secGrads);
    double computeRidgeSurface(vec3 *gradients, mat3 *hessians);
    double computeRidgeSurfaceTest(vec3 *gradients, mat3 *hessians);
    double computeRidge(Vector80d sampleVector);
    double computeRidge2D(Vector24d sampleVector);
};

#endif