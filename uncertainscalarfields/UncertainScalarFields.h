#ifndef __UncertainScalarFields_h
#define __UncertainScalarFields_h


#include "vtkPolyDataAlgorithm.h" //superclass
#include "vtkImageAlgorithm.h"
#include "vtkSmartPointer.h" // compiler errors if this is forward declared
#include "vtkPointSet.h"

class vtkPolyData;
class vtkImageData;
class vtkTransform;
class vtkInformation;
class vtkInformationVector;
class vtkIterativeClosestPointTransform;

using namespace std;

class UncertainScalarFields : public vtkImageAlgorithm
{
public:
	static UncertainScalarFields *New();
	vtkTypeMacro(UncertainScalarFields, vtkImageAlgorithm);
	void PrintSelf(ostream &os, vtkIndent indent);

	vtkSetMacro(numOfFields, int);
	vtkGetMacro(numOfFields, int);

	vtkSetMacro(theta, double);
	vtkGetMacro(theta, double);

	vtkSetMacro(isotro, bool);
    vtkGetMacro(isotro, bool);

	vtkSetMacro(base, double);
	vtkGetMacro(base, double);

	vtkSetVector3Macro(origin_grid, double);
	vtkGetVector3Macro(origin_grid, double);

	vtkSetVector3Macro(bounds_grid, double);
	vtkGetVector3Macro(bounds_grid, double);

	vtkSetVector3Macro(resolution_grid, double);
	vtkGetVector3Macro(resolution_grid, double);

	vtkSetVector2Macro(noise, double);
	vtkGetVector2Macro(noise, double);

	vtkSetMacro(shiftEQ, bool);
    vtkGetMacro(shiftEQ, bool);

    vtkSetMacro(shiftX, bool);
    vtkGetMacro(shiftX, bool);
	
    vtkSetMacro(shiftY, bool);
    vtkGetMacro(shiftY, bool);

	vtkSetMacro(shiftZ, bool);
    vtkGetMacro(shiftZ, bool);

	vtkSmartPointer<vtkImageData> inputGrid;
	vtkSmartPointer<vtkImageData> seedGrid;
	double* inputGridSpacing;
	int* inputGridDimensions;

protected:
	UncertainScalarFields();
	~UncertainScalarFields();
	unsigned int dim[3];

	// Make sure the pipeline knows what type we expect as input
	int FillInputPortInformation( int port, vtkInformation* info );
	int FillOutputPortInformation( int port, vtkInformation* info );
	int RequestInformation(vtkInformation *, vtkInformationVector **, vtkInformationVector *); //the function that makes this class work with the vtk pipeline

	// Generate output
	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *); //the function that makes this class work with the vtk pipeline
	int RequestUpdateExtent(vtkInformation*,vtkInformationVector** inputVector,vtkInformationVector* outputVector);

	double noise[2];
	double origin_grid[3];
	double bounds_grid[3];
	double resolution_grid[3];
	double domRange[3];

	int numOfFields;
	double theta;
	bool isotro;
	double base;
	bool shiftEQ;
	bool shiftX;
	bool shiftY;
	bool shiftZ;

	vtkSmartPointer<vtkImageData> data;
};
#endif
