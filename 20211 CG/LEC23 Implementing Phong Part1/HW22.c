#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define my_PI 3.141592
#define READBUFSIZ 70
#define X_min 0
#define X_Max 1
#define Y_min 2
#define Y_Max 3
#define Z_min 4
#define Z_Max 5

static char* vsSource = "#version 140 \n\
	in vec4 aPosition; \n\
	in vec4 aNormal; \n\
   	out vec4 vColor; \n\
	\n\
	uniform mat4 uscale; \n\
	uniform mat4 urotate;  \n\
	uniform mat4 utrans_o; \n\
	uniform mat4 utrans_z; \n\
	uniform mat4 um_view; \n\
	uniform mat4 um_persp; \n\
	\n\
	uniform vec4 l_pos; \n\
	uniform vec4 l_amb; \n\
	uniform vec4 l_dif; \n\
	uniform vec4 l_att; \n\
	\n\
	uniform vec4 m_amb; \n\
	uniform vec4 m_dif; \n\
	\n\
	void main(void) { \n\
		mat4 mModel		= utrans_z * uscale * urotate * utrans_o; \n\
  		vec4 vPosition	= mModel * aPosition; \n\
		\n\
		vec4 amb		= l_amb * m_amb; \n\
		mat4 mNormal	= transpose(inverse(mModel)); \n\
		vec4 vNormal	= mNormal * aNormal; \n\
		\n\
		vec3	n		= normalize(vNormal.xyz); \n\
		vec3	l		= normalize(l_pos.xyz - vPosition.xyz); \n\
		float	d		= length(l_pos.xyz - vPosition.xyz); \n\
		float	denom	= l_att.x + l_att.y * d + l_att.z * d * d; \n\
		vec4	dif		= max(dot(l, n), 0.0) * l_dif * m_dif / denom; \n\
		\n\
		gl_Position = vPosition; \n\
		vColor		= amb + dif; \n\
	}";

static char* fsSource = "#version 140 \n\
  	in vec4 vColor; \n\
	void main(void) { \n\
  		gl_FragColor = vColor; \n\
	}";

GLuint vs = 0;
GLuint fs = 0;
GLuint prog = 0;

char buf[1024];
int DRAW_MODE = 0, DRAW_LINE = 0, DRAW_ROTATE = 0;
float s=0.0, t = 0.0;

// read from file teapot.obj.txt
int num_vertices, num_faces;
GLfloat *vertices = NULL;
GLfloat *normals = NULL;
GLushort *indices = NULL;
GLfloat minMax[6] = {0.0}; // X_min, X_Max, Y_min, Y_Max, Z_min, Z_Max
GLfloat center[3] = {0.0}; // X, Y, Z
GLfloat boxlen[3] = {0.0}; // X, Y, Z
FILE *f;



void initVertices() {

	int i, j;
	char buf[READBUFSIZ];
	GLfloat temp;


	if ((f = fopen("teapot.obj", "r")) == NULL) {
		printf("failed to open teapot.obj\n");
		exit(1);
	}
	fgets(buf, READBUFSIZ, f);
	sscanf(buf, "%d %d", &num_vertices, &num_faces);


	vertices = (GLfloat*)malloc(num_vertices * 4 * sizeof(GLfloat));


	/*
	
	
		vertices


	*/
	fgets(buf, READBUFSIZ, f);
	sscanf(buf, "%f %f %f", &(vertices[0]), &(vertices[1]), &(vertices[2]));
	vertices[3] = 1.0;
	minMax[X_min] = vertices[0]; minMax[X_Max] = vertices[0];
	minMax[Y_min] = vertices[1]; minMax[Y_Max] = vertices[1];
	minMax[Z_min] = vertices[2]; minMax[Z_Max] = vertices[2];

	for (i = 1; i < num_vertices; i++) {
		fgets(buf, READBUFSIZ, f);
		sscanf(buf, "%f %f %f",	&(vertices[i*4+0]), &(vertices[i*4+1]), &(vertices[i*4+2]));
		vertices[i*4+3] = 1.0;
		// find min, Max
		temp = vertices[i*4+0];
		if 		(minMax[X_min] > temp)	minMax[X_min] = temp;
		else if (minMax[X_Max] < temp)	minMax[X_Max] = temp;
		temp = vertices[i*4+1];
		if 		(minMax[Y_min] > temp)	minMax[Y_min] = temp;
		else if (minMax[Y_Max] < temp)	minMax[Y_Max] = temp;
		temp = vertices[i*4+2];
		if		(minMax[Z_min] > temp)	minMax[Z_min] = temp;
		else if (minMax[Z_Max] < temp)	minMax[Z_Max] = temp;
	}
	// find center
	center[0] = (minMax[X_min] + minMax[X_Max]) / 2;
	center[1] = (minMax[Y_min] + minMax[Y_Max]) / 2;
	center[2] = (minMax[Z_min] + minMax[Z_Max]) / 2;
	// find boxlen
	boxlen[0] = minMax[X_Max] - minMax[X_min];
	boxlen[1] = minMax[Y_Max] - minMax[Y_min];
	boxlen[2] = minMax[Z_Max] - minMax[Z_min];
	/*
	
	
		normals

		
	*/
	normals = (GLfloat*)malloc(num_vertices * 4 * sizeof(GLfloat));
	for (i = 0; i < num_vertices; i++) {		
		fgets(buf, READBUFSIZ, f);
		sscanf(buf, "%f %f %f",
		 &normals[i*4+0], &normals[i*4+1], &normals[i*4+2]);		
		normals[i*4+3] = 1.0f;
	}
	printf("%f %f %f\n",
		 normals[0], normals[1], normals[2]);
	/*
	
	
		indices

		
	*/
	indices	= (GLushort*)malloc(3 * num_faces * sizeof(GLushort));
	for (i = 0; i < num_faces; i++) {
		fgets(buf, READBUFSIZ, f);
		sscanf(buf, "%hu %hu %hu", &(indices[i*3+0]), &(indices[i*3+1]), &(indices[i*3+2]));
	}
	fclose(f);
}

void myinit(void) {
	GLuint status;
	
	printf("***** Your student number and name *****\n");

	vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vsSource, NULL);
	glCompileShader(vs);
	glGetShaderiv(vs, GL_COMPILE_STATUS, &status);
	printf("vs compile status = %s\n", (status == GL_TRUE) ? "true" : "false");
	glGetShaderInfoLog(vs, sizeof(buf), NULL, buf);
	printf("vs log = [%s]\n", buf);
	 
	fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fsSource, NULL);
	glCompileShader(fs);  
	glGetShaderiv(fs, GL_COMPILE_STATUS, &status);
	printf("fs compile status = %s\n", (status == GL_TRUE) ? "true" : "false");
	glGetShaderInfoLog(fs, sizeof(buf), NULL, buf);
	printf("fs log = [%s]\n", buf);
	 
	prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);  
	glGetProgramiv(prog, GL_LINK_STATUS, &status);
	printf("program link status = %s\n", (status == GL_TRUE) ? "true" : "false");
	glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
	printf("link log = [%s]\n", buf);
	glValidateProgram(prog);
	glGetProgramiv(prog, GL_VALIDATE_STATUS, &status);
	printf("program validate status = %s\n", (status == GL_TRUE) ? "true" : "false");
	glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
	printf("validate log = [%s]\n", buf);
	glUseProgram(prog);  

	initVertices();
	printf("Center: (%f, %f, %f)\n", center[0], center[1], center[2]);
	printf("Box length in x-direction: %f\n", boxlen[0]);
	printf("Box length in y-direction: %f\n", boxlen[1]);
	printf("Box length in z-direction: %f\n", boxlen[2]);

	GLuint loc;
	GLuint vbo[1];
	int v_size = num_vertices*4*sizeof(GLfloat);
	// using vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);	
	glBufferData(GL_ARRAY_BUFFER, 2*num_vertices*4*sizeof(GLfloat), NULL, GL_STATIC_DRAW);

	glBufferSubData(GL_ARRAY_BUFFER, 0, num_vertices*4*sizeof(GLfloat), vertices);	
	loc = glGetAttribLocation(prog, "aPosition");
	glEnableVertexAttribArray(loc);
	glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
	
	glBufferSubData(GL_ARRAY_BUFFER, num_vertices*4*sizeof(GLfloat), num_vertices*4*sizeof(GLfloat), normals);
	loc = glGetAttribLocation(prog, "aNormal");
	glEnableVertexAttribArray(loc);
	glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)(num_vertices*4*sizeof(GLfloat)));

	free(vertices);
	free(normals);
	
	glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);
	glEnable(GL_DEPTH_TEST);

	
}
void mykeyboard(unsigned char key, int x, int y) {
	switch (key) {
	case 27: // ESCAPE
		exit(0);
		break;
	}
}
void mymenu(int id)
{
	if (id == -1) {
		DRAW_ROTATE = (DRAW_ROTATE + 1) % 2;
	}
	else if (id == 0)
		DRAW_MODE = 0;
	else if (id == 1)
		DRAW_MODE = 1;
	else if (id == 2)
		DRAW_MODE = 2;
	else if (id == 3)
		DRAW_LINE = 1;
	else if (id == 4)
		DRAW_LINE = 0;
	else if (id == 5) {
		DRAW_ROTATE = 0;
		DRAW_LINE = 0;
		t = 0.0;
	}

	if (DRAW_LINE == 1)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glutPostRedisplay();
}


void myidle(void) {
	if (DRAW_ROTATE == 1) {

		t += 0.006f;
		s += 0.006f;
	}
	// redisplay 
	glutPostRedisplay();
}

GLfloat m[16], m_view[16], m_ortho[16], m_persp[16];

void vec_minus(float *v3, float *v1, float *v2)
{   // v3 = v1 - v2
	for (int i = 0; i < 4; i++)
		v3[i] = v1[i] - v2[i];
}
float vec_dot_prod(float *v1, float *v2)
{   // return the dot product of v1 and v2
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

float vec_len(float *v)
{
	// return the length of v
	return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void vec_normalize(float *nv, float *v)
{   // nv = normalized v
	float len;
	len = vec_len(v);
	nv[0] = v[0] / len;
	nv[1] = v[1] / len;
	nv[2] = v[2] / len;
	nv[3] = v[3] / len;
}

void vec_cross_prod(float *v3, float *v1, float *v2)
{
	// v3 = v1 x v2
	v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
	v3[1] = - v1[0] * v2[2] + v1[2] * v2[0];
	v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

void vec_assign(float* v2, float *v1)
{
	for (int i = 0; i < 4; i++)
		v2[i] = v1[i];
}

void print_mat(float *m)
{
	for (int i = 0; i < 4; i++) {
		printf("%f  %f  %f   %f\n", m[i], m[i + 4], m[i + 8], m[i + 12]);
	}
}

void LookAt(float *m_view, float *eye, float *at, float *up)
{// construct m_view by using eye, at, up
	float u[4], v[4], n[4], p[4];
	vec_assign(p, eye);
	vec_minus(n, at, eye);
	vec_normalize(n, n);
	
	for (int i = 0; i < 4; i++)
		v[i] = -vec_dot_prod(up, n)*n[i] + up[i];
	vec_normalize(v, v);
	
	vec_cross_prod(u, n, v);
	vec_normalize(u, u);

	m_view[0] = u[0];    m_view[4] = u[1];    m_view[8] = u[2];    m_view[12] = -vec_dot_prod(p, u);
	m_view[1] = v[0];    m_view[5] = v[1];    m_view[9] = v[2];    m_view[13] = -vec_dot_prod(p, v);
	m_view[2] = n[0];    m_view[6] = n[1];    m_view[10] = n[2];   m_view[14] = -vec_dot_prod(p, n);
	m_view[3] = 0.0;     m_view[7] = 0.0;     m_view[11] = 0.0;    m_view[15] = 1.0;

//	printf("m_view:\n");
//	print_mat(m_view);
}
void OrthoProj (float* m_ortho, GLfloat xmin, GLfloat xmax, GLfloat ymin, GLfloat ymax, GLfloat zmin, GLfloat zmax) {
	GLfloat vec[4], uec[4], wec[4];
	int i;

	// orthogonal projection matrix
	m_ortho[0] = 2.0 / (xmax - xmin);	m_ortho[4] = 0.0;					m_ortho[8] = 0.0;					m_ortho[12] = -(xmax + xmin) / (xmax - xmin);
	m_ortho[1] = 0.0;					m_ortho[5] = 2.0 / (ymax - ymin);	m_ortho[9] = 0.0;					m_ortho[13] = -(ymax + ymin) / (ymax - ymin);
	m_ortho[2] = 0.0;					m_ortho[6] = 0.0; 					m_ortho[10] = 2.0 / (zmax - zmin);	m_ortho[14] = -(zmax + zmin) / (zmax - zmin);
	m_ortho[3] = 0.0;					m_ortho[7] = 0.0;					m_ortho[11] = 0.0;					m_ortho[15] = 1.0;


//	printf("m_ortho:\n");
//	print_mat(m_ortho);
}
void PerspProj(float* m_persp, GLfloat xmin, GLfloat xmax, GLfloat ymin, GLfloat ymax, GLfloat zmin, GLfloat zmax)
{
	m_persp[0] = 2 * zmin / (xmax - xmin);	m_persp[4] = 0;							m_persp[8] = -(xmax + xmin) / (xmax - xmin);	m_persp[12] = 0;
	m_persp[1] = 0;							m_persp[5] = 2 * zmin / (ymax - ymin);	m_persp[9] = -(ymax + ymin) / (ymax - ymin);	m_persp[13] = 0;
	m_persp[2] = 0;							m_persp[6] = 0;							m_persp[10] = (zmax + zmin) / (zmax - zmin);	m_persp[14] = -2 * zmax * zmin / (zmax - zmin);
	m_persp[3] = 0;							m_persp[7] = 0;							m_persp[11] = 1;								m_persp[15] = 0;
}
void myFrustum(float *m, GLfloat xm, GLfloat xM, GLfloat ym, GLfloat yM, GLfloat zm, GLfloat zM)
{
	if (xm < xM && ym < yM && zm > 0 && zm < zM) {
		GLfloat xlen = xM - xm;
		GLfloat ylen = yM - ym;
		GLfloat zlen = zM - zm;
		m[0] = 2*zm/xlen;m[4] = 0.0;       m[8] = -(xM+xm)/xlen;    m[12] = 0.0;
		m[1] = 0.0;      m[5] = 2*zm/ylen; m[9] = -(yM+ym)/ylen;    m[13] = 0.0;
		m[2] = 0.0;      m[6] = 0.0;       m[10] = (zM+zm)/zlen;	m[14] = -2*zM*zm/zlen;
		m[3] = 0.0;      m[7] = 0.0;       m[11] = 1.0;				m[15] = 0.0;
	}
	else {
		m[0] = 1.0;		 m[4] = 0.0;       m[8] = 0.0;			    m[12] = 0.0;
		m[1] = 0.0;      m[5] = 1.0;	   m[9] = 0.0;    			m[13] = 0.0;
		m[2] = 0.0;      m[6] = 0.0;       m[10] = 1.0;				m[14] = 0.0;
		m[3] = 0.0;      m[7] = 0.0;       m[11] = 0.0;				m[15] = 1.0;
	}
}
void mat_translate(float *m, float* t_v)
{
	m[0] = 1.0;      m[4] = 0.0;     m[8] = 0.0;      m[12] = t_v[0];
	m[1] = 0.0;      m[5] = 1.0;     m[9] = 0.0;      m[13] = t_v[1];
	m[2] = 0.0;      m[6] = 0.0;     m[10] = 1.0;     m[14] = t_v[2];
	m[3] = 0.0;      m[7] = 0.0;     m[11] = 0.0;     m[15] = 1.0;
}
void mat_scale(float *m, float *s_v)
{
	m[0] = s_v[0];   m[4] = 0.0;     m[8] = 0.0;      m[12] = 0.0;
	m[1] = 0.0;      m[5] = s_v[1];  m[9] = 0.0;      m[13] = 0.0;
	m[2] = 0.0;      m[6] = 0.0;     m[10] = s_v[2];  m[14] = 0.0;
	m[3] = 0.0;      m[7] = 0.0;     m[11] = 0.0;     m[15] = 1.0;
}
void mat_rotate_x(float *m, float t)
{
	m[0] = 1.0;		m[4] = 0.0;		m[8] = 0.0;			m[12] = 0.0;
	m[1] = 0.0;		m[5] = cos(t);  m[9] = -sin(t);		m[13] = 0.0;
	m[2] = 0.0;		m[6] = sin(t);  m[10] = cos(t);		m[14] = 0.0;
	m[3] = 0.0;		m[7] = 0.0;     m[11] = 0.0;		m[15] = 1.0;
}

void mat_rotate_y(float *m, float t)
{
	m[0] = cos(t);   m[4] = 0.0;     m[8] = sin(t);   m[12] = 0.0;
	m[1] = 0.0;      m[5] = 1.0;     m[9] = 0.0;      m[13] = 0.0;
	m[2] = -sin(t);  m[6] = 0.0;     m[10] = cos(t);  m[14] = 0.0;
	m[3] = 0.0;      m[7] = 0.0;     m[11] = 0.0;     m[15] = 1.0;
}

void mat_rotate_z(float *m, float t)
{
	m[0] = cos(t); m[4] = -sin(t); m[8] = 0.0;  m[12] = 0.0;
	m[1] = sin(t); m[5] = cos(t);  m[9] = 0.0;  m[13] = 0.0;
	m[2] = 0.0;    m[6] = 0.0;     m[10] = 1.0; m[14] = 0.0;
	m[3] = 0.0;    m[7] = 0.0;     m[11] = 0.0; m[15] = 1.0;
}
void mat_identity(float *m)
{
	m[0] = 1.0;   m[4] = 0.0;  m[8] = 0.0;   m[12] = 0.0;
	m[1] = 0.0;   m[5] = 1.0;  m[9] = 0.0;   m[13] = 0.0;
	m[2] = 0.0;   m[6] = 0.0;  m[10] = 1.0;  m[14] = 0.0;
	m[3] = 0.0;   m[7] = 0.0;  m[11] = 0.0;  m[15] = 1.0;
}

void setLightAndMaterial(void)
{
	GLfloat l_pos[4] = {2.0, 2.0, -2.0, 1.0};
	GLfloat l_amb[4] = {0.6, 0.6, 0.6, 1.0};
	GLfloat l_dif[4];
	GLfloat l_att[4] = {1.0, 0.1, 0.0, 1.0};
	GLfloat m_amb[4];
	GLfloat m_dif[4] = {0.8, 0.8, 1.0, 1.0};
	GLuint loc;
	l_dif[0] = 255.0/255.0;
	l_dif[1] = 234.0/255.0;
	l_dif[2] = 238.0/255.0;
	l_dif[3] = 1.0;
	m_amb[0] = 127.0/255.0;
	m_amb[1] = 190.0/255.0;
	m_amb[2] = 235.0/255.0;
	m_amb[3] = 1.0;

	loc = glGetUniformLocation(prog, "l_pos");
	glUniform4fv(loc, 1, l_pos);
	loc = glGetUniformLocation(prog, "l_amb");
	glUniform4fv(loc, 1, l_amb);
	loc = glGetUniformLocation(prog, "l_dif");
	glUniform4fv(loc, 1, l_dif);
	loc = glGetUniformLocation(prog, "l_att");
	glUniform4fv(loc, 1, l_att);
	loc = glGetUniformLocation(prog, "m_amb");
	glUniform4fv(loc, 1, m_amb);
	loc = glGetUniformLocation(prog, "m_dif");
	glUniform4fv(loc, 1, m_dif);
}

void mydisplay(void) {
	GLuint loc;
	glClearColor(0.2f, 0.2f, 0.2f, 1.0f); // gray
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GLfloat maxlen = boxlen[0];
	if (maxlen < boxlen[1]) maxlen = boxlen[1];
	if (maxlen < boxlen[2]) maxlen = boxlen[2];

	float v_scale[3];
	float v_trans[3];
	float s = sqrt(pow(boxlen[0], 2.0)+pow(boxlen[1], 2.0)+pow(boxlen[2], 2.0));
	for (int i = 0; i < 3; i++) {
		v_scale[i] = 1.2 / s;
		v_trans[i] = -center[i];
	}
	mat_scale(m, v_scale);
	loc = glGetUniformLocation(prog, "uscale");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m);

	if (DRAW_MODE == 0) 
		mat_rotate_x(m, t);
	else if (DRAW_MODE == 1) 
		mat_rotate_y(m, t);
	else if (DRAW_MODE == 2) 
		mat_rotate_z(m, t);

	loc = glGetUniformLocation(prog, "urotate");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m);
	
	mat_translate(m, v_trans);
	loc = glGetUniformLocation(prog, "utrans_o");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m);

	mat_identity(m);
	m[14] = -0.5; // trans z
	loc = glGetUniformLocation(prog, "utrans_z");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m);

	float eye[4] = { 0.0, 0.0, 0.0, 1.0 },
		  at[4] = { 0.0, 0.0, 1.0, 1.0 },
		  up[4] = { 0.0, 1.0, 0.0, 0.0 };

	//LookAt(m_view, eye, at, up);
	mat_identity(m_view);
	loc = glGetUniformLocation(prog, "um_view");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m_view);
	//float xmin = -0.7, xmax = +0.5, ymin = -0.5, ymax = +0.5, zmin = 1.8, zmax = +4.0;


	//myFrustum(m_persp, xmin, xmax, ymin, ymax, zmin, zmax);
	mat_identity(m_persp);
	loc = glGetUniformLocation(prog, "um_persp");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m_persp);

	setLightAndMaterial();
	   	 
	glDrawElements(GL_TRIANGLES, num_faces * 3, GL_UNSIGNED_SHORT, indices);
	glFlush();

	glutSwapBuffers();
}


int main(int argc, char* argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("*** Your Student Number and Name ***");
	glutDisplayFunc(mydisplay);
	glutIdleFunc(myidle);
	glutKeyboardFunc(mykeyboard);
	glutCreateMenu(mymenu);

	glutAddMenuEntry("-----", 100);
	glutAddMenuEntry("Rotate", -1);
	glutAddMenuEntry("x-axis", 0);
	glutAddMenuEntry("y-axis", 1);
	glutAddMenuEntry("z-axis", 2);
	glutAddMenuEntry("-----", 100);
	glutAddMenuEntry("Draw by line", 3);
	glutAddMenuEntry("Draw by Polygon", 4);
	glutAddMenuEntry("-----", 100);
	glutAddMenuEntry("Initialize", 5);


	glutAttachMenu(GLUT_RIGHT_BUTTON);
	glewInit();
	myinit();
	glutMainLoop();
	return 0;
}