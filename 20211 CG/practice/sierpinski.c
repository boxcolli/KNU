#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SPSK_TRIAL  5000
#define CO_X        0
#define CO_Y        1
#define LI_M        0
#define LI_N        1
#define RANDF_DIGIT 3 // MAX 4

const static char* vsSource = "#version 120 \n\
    attribute vec4 vert; \n\
    void main(void) { \n\
        gl_Position = vert; \n\
    }";
const static char* fsSource = "#version 120 \n\
    void main(void) {\n\
        gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0); \n\
    }";
float bg[4] = { 124.0 / 255.0, 124.0 / 255.0, 124.0 / 255.0, 1.0 };
GLuint vs = 0;
GLuint fs = 0;
GLuint prog = 0;
GLfloat *triangle[3];
GLfloat vertices[SPSK_TRIAL * 4];

void myinit();
void mykeyboard(unsigned char key, int x, int y);
void mydisplay();
void spsk(float *t[], float *v, int trial);
float randf(float a, float b, int digit);

int main(int argc, char* argv[])
{
	printf("sierpinski with %d trials\n", SPSK_TRIAL);

	triangle[0] = (GLfloat *)malloc(4 * sizeof(GLfloat));
	triangle[1] = (GLfloat *)malloc(4 * sizeof(GLfloat));
	triangle[2] = (GLfloat *)malloc(4 * sizeof(GLfloat));
	triangle[0][0] = -0.5; triangle[0][1] = -0.5; triangle[0][2] = 0.0; triangle[0][3] = 1.0;
	triangle[1][0] = +0.5; triangle[1][1] = -0.5; triangle[1][2] = 0.0; triangle[1][3] = 1.0;
	triangle[2][0] = -0.5; triangle[2][1] = +0.5; triangle[2][2] = 0.0; triangle[2][3] = 1.0;
	for (int i = 0; i < 3; i++) {
		printf("%f %f %f %f\n", triangle[i][0], triangle[i][1], triangle[i][2], triangle[i][3]);
	}
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("2016112905 김민섭");
	glutDisplayFunc(mydisplay);
	glutKeyboardFunc(mykeyboard);
	glewInit();
	myinit();
	glutMainLoop();
	return 0;
}

void myinit()
{
	vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vsSource, NULL);
	glCompileShader(vs);

	fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fsSource, NULL);
	glCompileShader(fs);

	prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);
	glUseProgram(prog);

	srand(time(NULL));
	spsk(triangle, vertices, SPSK_TRIAL);
}
void mykeyboard(unsigned char key, int x, int y)
{
	switch (key) {
	case 27: //ESCAPE
		exit(0);
		break;
	}
}
void mydisplay()
{
	GLuint loc;
	glClearColor(randf(0.0, 255.0, 3), randf(0.0, 255.0, 3), randf(0.0, 255.0, 3), randf(0.0, 255.0, 3));
	glClear(GL_COLOR_BUFFER_BIT);
	loc = glGetAttribLocation(prog, "vert");
	glEnableVertexAttribArray(loc);
	glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, vertices);
	glDrawArrays(GL_POINTS, 0, SPSK_TRIAL);
	glFlush();
}

void spsk(float *t[], float *v, int trial)
{
	/* Sierpinski Gasket Algorithm */

	float line1[2], line2[2];
	// LI_M: m from (y = mx + n)
	// LI_N: n from (y = mx + n)

	float A[2], p[2], q[2];
	// CO_X: x coordinate
	// CO_Y: y coordinate

	int i, trand0, trand1, vrand;

	/* 1. p = random point inside of the triangle */
	// ( choose a random point A on the line v0-v1 )
	// ! potential error: divide by 0 !
	if (t[0][CO_X] - t[1][CO_X] != 0) {
		line1[LI_M] = (t[0][CO_Y] - t[1][CO_Y]) / (t[0][CO_X] - t[1][CO_X]);
		line1[LI_N] = t[0][CO_Y] - line1[LI_M] * t[0][CO_X];
		printf("line1 (%6.3f, %6.3f)\n", line1[LI_M], line1[LI_N]);
		A[CO_X]     = randf(t[0][CO_X], t[1][CO_X], RANDF_DIGIT);
		A[CO_Y]     = line1[LI_M] * A[CO_X] + line1[LI_N];
	}
	else {
		printf("line1 (infinity, )\n");
		A[CO_X] = t[0][CO_X];
		A[CO_Y] = randf(t[0][CO_Y], t[1][CO_Y], RANDF_DIGIT);
	}	
	printf("randA (%6.3f, %6.3f)\n", A[CO_X], A[CO_Y]);

	// ( choose a random point p on the line A-v2 )
	if (A[CO_X] - t[2][CO_X] != 0) {
		line2[LI_M] = (A[CO_Y] - t[2][CO_Y]) / (A[CO_X] - t[2][CO_X]);
		line2[LI_N] = A[CO_Y] - line2[0] * A[CO_X];
		printf("line2 (%6.3f, %6.3f)\n", line2[LI_M], line2[LI_N]);
		p[CO_X]     = randf(A[CO_X], t[2][CO_X], RANDF_DIGIT);
		p[CO_Y]     = line2[LI_M] * p[CO_X] + line2[LI_N];
	}
	else {
		printf("line2 (infinity, )\n");
		p[CO_X] = A[CO_X];
		p[CO_Y] = randf(A[CO_Y], t[2][CO_Y], RANDF_DIGIT);
	}
	printf("randp (%6.3f, %6.3f)\n", p[CO_X], p[CO_Y]);

	for (i = 0; i < trial; i++) {
		/* 2. vrand = random vertex of triangle */
		vrand = rand() % 3;

		/* 3. find q = (p+v)/2 */
		q[CO_X] = (p[CO_X] + t[vrand][CO_X]) / 2.0;
		q[CO_Y] = (p[CO_Y] + t[vrand][CO_Y]) / 2.0;

		/* 4. write q */
		*(v + i * 4 + 0) = q[0];
		*(v + i * 4 + 1) = q[1];
		*(v + i * 4 + 2) = 0.0;
		*(v + i * 4 + 3) = 1.0;

		/* 5. next p is q */
		p[CO_X] = q[CO_X];
		p[CO_Y] = q[CO_Y];

		/* 6. repeat from 2 */
		//printf("spsk trial %dth\n", i);
	}
	printf("spsk done\n");
}

float randf(float A, float B, int digit)
{
	int i;
	float result;

	// confirm A < B
	if (A > B) {
		result = A; // using result as temp
		A = B;
		B = result;
	}
	// make floats big
	for (i = 0; i < digit; i++) {
		A *= 10;
		B *= 10;
	}
	// choose rand from range(min, max)
	if (B > 0.0)
		result = (float)(rand() % ((int)B - (int)A + 1) + (int)A);
	else
		result = (float)(rand() % -((int)B - (int)A + 1) + (int)A);

	// make result small
	for (i = 0; i < digit; i++)
		result /= 10;
	return result;
}