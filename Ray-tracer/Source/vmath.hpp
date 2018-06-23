#include <cmath>

#if defined __linux__ || defined __APPLE__
// "Compiled for Linux
#else
// Windows doesn't define these values by default, Linux does
#define M_PI 3.141592653589793
#endif

// Define the ASSERT and DEBUG macros
#ifdef _WIN32
#define ASSERT _ASSERT
#define assert _ASSERT
#ifdef _DEBUG 
#define DEBUG
#endif
#endif


#ifdef DEBUG
#define LOGINT(x) std::cout << x << std::endl;
#define LOGF(x) std::cout << x << std::endl;
#define LOGV2(Vector2) std::cout << Vector2.x << ", " << Vector2.y << std::endl; 
#define LOGV3(Vector3) std::cout << Vector3.x << ", " << Vector3.y << ", " << Vector3.z << std::endl; 
#define LOG(macro_format, ...)										\
{																	\
	printf(macro_format, ##__VA_ARGS__);							\
    const size_t macro_len = strlen(macro_format);					\
    if(macro_len > 0 && macro_format[macro_len - 1] != '\n')		\
	{																\
        printf("\n");												\
    }																\
}

#else
#define LOG(...) { }
#endif


namespace vmath
{
	typedef unsigned int uint;
	typedef unsigned long ulong;
	typedef unsigned short ushort;
	typedef unsigned char uchar;

	const float Pi = 3.1415926535897932384626433832795f;
	const float TwoPi = Pi * 2;
	const float HalfPi = Pi / 2;
	const float QuarterPi = Pi / 4;

	inline float DegToRad(float deg) { return deg * Pi / 180.0f; }

	/// Returns a random float between zero and 1
	inline float RandFloat() { return static_cast<float>((rand()) / (RAND_MAX + 1.0)); }

	/// Returns a random float between x and y
	inline float RandInRange(float x, float y) { return x + RandFloat()*(y - x); }

	/// Returns a random int between from and to
	inline int RandInRange(int from, int to) { return from + rand() % (to - from); }

	template <class T>
	inline T Modulo(T x, T m) { return (x % m + m) % m; }

	template <class T>
	T Lerp(T a, T b, float t)
	{
		if (t < 0)
			t = 0;
		else if (t > 1)
			t = 1;
		//
		return a * (1.0f - t) + b * t;
	}

	template <class T>
	void ClampDown(T compare, T min, T equals)
	{
		if (compare < min)
		{
			compare = 0;
		}
	}

	///
	/// 2d vector
	///
	struct Vector2
	{
		union
		{
			// Holds all the values 
			float f[2];

			struct
			{
				/// Holds the value along the x axis
				float x;

				/// Holds the value along the y axis
				float y;
			};
		};
		/// The default constructor creates a zero vector.
		Vector2();

		/// Creates a vector with the given components
		Vector2(float x, float y);

		void UpdateArray();


		/// Returns the value of the given vector added to this
		Vector2 operator+(const Vector2& v) const;

		/// Returns the value of the given vector subtracted from this
		Vector2 operator-(const Vector2& v) const;

		/// Returns a copy of this vector scaled the given value
		Vector2 operator*(const float value) const;

		/// Returns a copy of this vector scaled the inverse of the value
		Vector2 operator/(const float value) const;

		/// Returns the negative this vector
		Vector2 operator-() const;

		/// Calculates and returns the dot product of this
		/// with the given vector
		float operator*(const Vector2& vector) const;

		/// Adds the given vector to this
		void operator+=(const Vector2& v);

		/// Subtracts the given vector from this
		void operator-=(const Vector2& v);

		/// Multiplies this vector by the given scalar
		void operator*=(const float value);

		/// Calculates and returns the dot product of this vector
		/// with the given vector
		float Dot(const Vector2& vector) const;

		///  Gets the magnitude of this vector
		float Magnitude() const;

		///  Gets the squared magnitude of this vector
		float SquareMagnitude() const;

		/// Turns a non-zero vector into a vector of unit length
		void Normalize();

		float Cross();

		/// Checks if the two vectors have identical components
		bool operator==(const Vector2& other) const;

		/// Checks if the two vectors have non-identical components
		bool operator!=(const Vector2& other) const;

		/// Zero all the components of the vector
		void Clear();

		Vector2 Slerp(float fact, const Vector2& r) const;

		static float Magnitude(const Vector2& v1, const Vector2& v2);

	};

	///
	/// 3d vector
	///
	struct Vector3
	{
		union
		{
			// Holds all the values 
			float f[3];

			struct
			{
				/// Holds the value along the x axis
				float x;

				/// Holds the value along the y axis
				float y;

				/// Holds the value along the z axis
				float z;
			};
		};

		/// The default constructor creates a zero vector.
		Vector3();

		/// Creates a vector with the given components
		Vector3(float x, float y, float z);

		void UpdateArray();


		/// Returns the value of the given vector added to this
		Vector3 operator+(const Vector3& v) const;

		/// Returns the value of the given vector subtracted from this
		Vector3 operator-(const Vector3& v) const;

		/// Returns a copy of this vector scaled the given value
		Vector3 operator*(const float value) const;

		/// Returns a copy of this vector scaled the inverse of the value
		Vector3 operator/(const float value) const;

		/// Returns the negative this vector
		Vector3 operator-() const;

		/// Calculates and returns the dot product of this
		/// with the given vector
		float operator *(const Vector3& vector) const;

		/// Adds the given vector to this
		void operator+=(const Vector3& v);

		/// Subtracts the given vector from this
		void operator-=(const Vector3& v);

		/// Multiplies this vector by the given scalar
		void operator*=(const float value);

		/// Calculates and returns the cross product of this vector
		/// with the given vector
		Vector3 Cross(const Vector3& vector) const;

		/// Calculates and returns the dot product of this vector
		/// with the given vector
		float Dot(const Vector3& vector) const;

		///  Gets the magnitude of this vector
		float Magnitude() const;

		///  Gets the squared magnitude of this vector
		float SquareMagnitude() const;

		/// Turns a non-zero vector into a vector of unit length
		void Normalize();


		/// Checks if the two vectors have identical components
		bool operator==(const Vector3& other) const;

		/// Checks if the two vectors have non-identical components
		bool operator!=(const Vector3& other) const;

		/// Zero all the components of the vector
		void Clear();

		Vector3 Slerp(float fact, const Vector3& r) const;


	};


	///4d vector
	struct Vector4
	{
		union
		{
			// Holds all the values 
			float f[4];

			struct
			{
				/// Holds the value along the x axis
				float x;

				/// Holds the value along the y axis
				float y;

				/// Holds the value along the z axis
				float z;

				float w;
			};
		};
		/// The default constructor creates a zero vector.
		Vector4();

		/// Creates a vector with the given components
		Vector4(float x, float y, float z, float w);

		void UpdateArray();


		/// Returns the value of the given vector added to this
		Vector4 operator+(const Vector4& v) const;

		/// Returns the value of the given vector subtracted from this
		Vector4 operator-(const Vector4& v) const;

		/// Returns a copy of this vector scaled the given value
		Vector4 operator*(const float value) const;

		/// Returns a copy of this vector scaled the inverse of the value
		Vector4 operator/(const float value) const;

		/// Returns the negative this vector
		Vector4 operator-() const;

		/// Calculates and returns the dot product of this
		/// with the given vector
		float operator *(const Vector4& vector) const;

		/// Adds the given vector to this
		void operator+=(const Vector4& v);

		/// Subtracts the given vector from this
		void operator-=(const Vector4& v);

		/// Multiplies this vector by the given scalar
		void operator*=(const float value);

		/* Cross product of a 4D vector's isn't feasible (although, if needed - to be implemented )
		/// Calculates and returns the cross product of this vector
		/// with the given vector
		Vector4 Cross(const Vector4& vector) const;
		*/

		/// Calculates and returns the dot product of this vector
		/// with the given vector
		float Dot(const Vector4& vector) const;

		///  Gets the magnitude of this vector
		float Magnitude() const;

		///  Gets the squared magnitude of this vector
		float SquareMagnitude() const;

		/// Turns a non-zero vector into a vector of unit length
		void Normalize();


		/// Checks if the two vectors have identical components
		bool operator==(const Vector4& other) const;

		/// Checks if the two vectors have non-identical components
		bool operator!=(const Vector4& other) const;

		/// Zero all the components of the vector
		void Clear();

		Vector4 Slerp(float fact, const Vector4& r) const;


	};


	/// Multiplication with Rhs Vector
	inline Vector2 operator*(float val, const Vector2& rhs)
	{
		return rhs * val;
	}

	/// Multiplication with Rhs Vector
	inline Vector3 operator*(float val, const Vector3& rhs)
	{
		return rhs * val;
	}

	/// Multiplication with Rhs Vector
	inline Vector4 operator*(float val, const Vector4& rhs)
	{
		return rhs * val;
	}

	Vector2 ToVector2(const Vector3& vec);
	Vector3 ToVector3(const Vector2& vec, float z);
	Vector3 ToVector3(const Vector2& vec);

	struct Matrix44
	{
		union
		{
			float m[4][4];
			float f[16];
			struct
			{
				Vector3 xAxis;
				float wx;
				Vector3 yAxis;
				float wy;
				Vector3 zAxis;
				float wz;
				/* If needed - fix the method in the cpp file
				Vector3 xAxisH;
				float wxh;
				Vector3 yAxisH;
				float wyh;
				Vector3 zAxisH;
				float wzh;
				*/
				Vector3 translation;
				float one;
			};
		};

		/// Construct a new matrix from explicit values
		Matrix44(float m00, float m01, float m02, float m03,
			float m10, float m11, float m12, float m13,
			float m20, float m21, float m22, float m23,
			float m30, float m31, float m32, float m33);

		/// Construct a new identity matrix
		Matrix44();


		//Split's up the matrix in vectors , to make finding the determinant waay easier(and slower)
		float Determinant33(float m11, float m12, float m13, float m21, float m22, float m23, float m31, float m32, float m33) const;

		/// Transform the given vector by this matrix.
		///
		/// @param vec vector that is asssumed to be homogenuos with w=1
		/// @return Resulting vector is asssumed to be homogenuos with w=1
		Vector3 operator*(const Vector3& vec) const;

		/// Matrix addition
		///
		/// @param mat Right side operand
		Matrix44 operator+(const Matrix44& mat) const;

		/// Matrix substraction
		///
		/// @param mat Right side operand
		Matrix44 operator-(const Matrix44& mat) const;

		/// Matrix multiplication
		///
		/// @param mat Right side operand
		Matrix44 operator*(const Matrix44& mat) const;

		/// Translation bit of the matrix
		Vector3 GetTranslation() const;

		/// Set the transltion of the matrix
		//
		void SetTranslation(const Vector3& vec);

		/// Get the x orientation axis 
		Vector3 GetXAxis() const;

		/// Get the y orientation axis 
		Vector3 GetYAxis() const;

		/// Get the z orientation axis 
		Vector3 GetZAxis() const;

		/// Get the determinant of this matrix
		float Determinant() const;

		/// Inverts this matrix
		bool Invert();

		/// Transposes this matrix
		void Transpose();


		/// Sets the orientation of the matrix to the orthogonal basis vector
		/// It perfoms no checks on the orthogonality!
		///
		/// @param x X orthogonal basis vector
		/// @param y Y orthogonal basis vector
		/// @param z Z orthogonal basis vector
		//
		void SetOrientation(const Vector3& x,
			const Vector3& y,
			const Vector3& z);

		/// Set orientation using Euler angles. Broken at current!
		void SetEulerAxis(float yaw, float pitch, float roll);

		/// Creates an identity matrix
		///
		/// @return Identity matrix
		//
		static Matrix44 CreateIdentity();

		/// Creates a transation matrix
		///
		/// @return Translation matrix
		static Matrix44 CreateTranslation(float x, float y, float z);

		static Matrix44 CreateScale(Vector3 scale);

		/// Creates a rotation matrix around an arbitrary axis
		static Matrix44 CreateRotate(float angle, const Vector3& axis);

		/// Angle in radians
		static Matrix44 CreateRotateX(float angle);

		/// Angle in radians
		static Matrix44 CreateRotateY(float angle);

		/// Angle in radians
		static Matrix44 CreateRotateZ(float angle);

		/// Creates an orthographic projection matrix
		static Matrix44 CreateOrtho(float left, float right, float bottom, float top, float nearZ, float farZ);

		/// Creates a frustum projection matrix
		static Matrix44 CreateFrustum(float left, float right, float bottom, float top, float nearZ, float farZ);

		/// Creates a perspective projection matrix from camera settings
		//
		static Matrix44 CreatePerspective(float fovy, float aspect, float nearZ, float farZ);

		/// Creates a look at matrix, usualy a view matrix  
		//
		static Matrix44 CreateLookAt(const Vector3& eye, const Vector3& center, const Vector3& up);

		/// Transfrom just the direction
		Vector3 TransformDirectionVector(const Vector3& direction);
	};

	Vector2::Vector2()
	{
		x, y = 0;
		//UpdateArray();
	}

	Vector2::Vector2(float x, float y)
	{
		this->x = x;
		this->y = y;
		//UpdateArray();
	}

	void Vector2::UpdateArray()
	{
		f[0] = x;
		f[1] = y;
	}

	Vector2 Vector2::operator+(const Vector2 & v) const
	{
		Vector2 vector;
		vector.x = this->x + v.x;
		vector.y = this->y + v.y;

		return vector;
	}

	Vector2 Vector2::operator-(const Vector2 & v) const
	{
		Vector2 vector;
		vector.x = this->x - v.x;
		vector.y = this->y - v.y;

		return vector;
	}

	Vector2 Vector2::operator*(const float value) const
	{
		Vector2 vector;
		vector.x = this->x * value;
		vector.y = this->y * value;

		return vector;
	}

	Vector2 Vector2::operator/(const float value) const
	{
		Vector2 vector;
		vector.x = this->x / value;
		vector.y = this->y / value;

		return vector;
	}

	Vector2 Vector2::operator-() const
	{
		Vector2 vector;
		vector.x = this->x * -1;
		vector.y = this->y * -1;

		return vector;
	}

	float Vector2::operator*(const Vector2 & vector) const
	{
		return (this->x * vector.x) + (this->y * vector.y);
	}

	void Vector2::operator+=(const Vector2 & v)
	{
		this->x += v.x;
		this->y += v.y;
	}

	void Vector2::operator-=(const Vector2 & v)
	{
		this->x -= v.x;
		this->y -= v.y;
	}

	void Vector2::operator*=(const float value)
	{
		this->x *= value;
		this->y *= value;
	}

	float Vector2::Dot(const Vector2 & vector) const
	{
		return (this->x * vector.x) + (this->y * vector.y);
	}

	float Vector2::Magnitude() const
	{
		return sqrt((x * x) + (y * y));
	}

	float Vector2::SquareMagnitude() const
	{
		return (pow(this->x, 2) + pow(this->y, 2));
	}

	void Vector2::Normalize()
	{
		float magnitude = Magnitude();
		if (magnitude > 0)
		{
			this->x /= magnitude;
			this->y /= magnitude;
		}
	}

	float Vector2::Cross()
	{
		return this->y - this->x;
	}

	bool Vector2::operator==(const Vector2 & other) const
	{
		if (this->x == other.x && this->y == other.y)
			return true;
		else
			return false;
	}



	bool Vector2::operator!=(const Vector2 & other) const
	{
		if (this->x == other.x && this->y == other.y)
			return false;
		else
			return true;
	}

	void Vector2::Clear()
	{
		this->x = 0;
		this->y = 0;

		for (auto i = 0; i < 2; i++)
		{
			f[i] = 0;
		}
	}

	Vector2 Vector2::Slerp(float fact, const Vector2 & r) const
	{
		Vector2 v0 = *this;
		// Dot product - the cosine of the angle between 2 vectors.
		float dot = Dot(r);

		const double DOT_THRESHOLD = 0.9995;
		if (dot > DOT_THRESHOLD)
		{
			// If the inputs are too close for comfort, linearly interpolate
			// and normalize the result.

			Vector2 result = v0 + fact*(r - v0);
			result.Normalize();
			return result;
		}

		// Clamp it to be in the range of Acos()
		if (dot < -1.0f)
			dot = -1.0f;
		else if (dot > 1.0f)
			dot = 1.0f;

		// Acos(dot) returns the angle between start and end,
		// And multiplying that by percent returns the angle betweend
		// start and the final result.
		float theta_0 = acos(dot);
		float theta = acos(dot) * fact;

		Vector2 RelativeVec = r - v0*dot;
		RelativeVec.Normalize();     // Orthonormal basis

		return v0 * cos(theta) + RelativeVec * sin(theta);
	}

	float Vector2::Magnitude(const Vector2 & v1, const Vector2 & v2)
	{
		//return v1.Magnitude() + v2.Magnitude();
		return sqrt(v1.x * v1.x + v1.y * v1.y + v2.x * v2.x + v2.y * v2.y);
	}


	Vector3::Vector3()
	{
		x, y, z = 0;
		//	UpdateArray();
	}

	Vector3::Vector3(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		//	UpdateArray();
	}

	void Vector3::UpdateArray()
	{
		f[0] = x;
		f[1] = y;
		f[2] = z;
	}

	Vector3 Vector3::operator+(const Vector3 & v) const
	{
		Vector3 vector;
		vector.x = this->x + v.x;
		vector.y = this->y + v.y;
		vector.z = this->z + v.z;

		return vector;
	}

	Vector3 Vector3::operator-(const Vector3 & v) const
	{
		Vector3 vector;
		vector.x = this->x - v.x;
		vector.y = this->y - v.y;
		vector.z = this->z - v.z;

		return vector;
	}

	Vector3 Vector3::operator*(const float value) const
	{
		Vector3 vector;
		vector.x = this->x * value;
		vector.y = this->y * value;
		vector.z = this->z * value;

		return vector;
	}

	Vector3 Vector3::operator/(const float value) const
	{
		Vector3 vector;
		vector.x = this->x / value;
		vector.y = this->y / value;
		vector.z = this->z / value;

		return vector;
	}

	Vector3 Vector3::operator-() const
	{
		Vector3 vector;
		vector.x = this->x * -1;
		vector.y = this->y * -1;
		vector.z = this->z * -1;

		return vector;
	}

	float Vector3::operator*(const Vector3 & vector) const
	{
		return (this->x * vector.x) + (this->y * vector.y) + (this->z + vector.z);
	}

	void Vector3::operator+=(const Vector3 & v)
	{
		this->x += v.x;
		this->y += v.y;
		this->z += v.z;
	}

	void Vector3::operator-=(const Vector3 & v)
	{
		this->x -= v.x;
		this->y -= v.y;
		this->z -= v.z;
	}

	void Vector3::operator*=(const float value)
	{
		this->x *= value;
		this->y *= value;
		this->z *= value;
	}

	Vector3 Vector3::Cross(const Vector3 & vector) const
	{
		Vector3 vec;

		vec.x = (this->y * vector.z) - (this->z * vector.y);
		vec.y = (this->z * vector.x) - (this->x * vector.z);
		vec.z = (this->x * vector.y) - (this->y * vector.x);

		return vec;
	}

	float Vector3::Dot(const Vector3 & vector) const
	{
		return  (this->x * vector.x) + (this->y * vector.y) + (this->z * vector.z);
	}

	float Vector3::Magnitude() const
	{
		return sqrt((x * x) + (y * y) + (z * z));
	}

	float Vector3::SquareMagnitude() const
	{
		return (pow(this->x, 2) + pow(this->y, 2) + pow(this->z, 2));
	}

	void Vector3::Normalize()
	{
		float magnitude = Magnitude();
		if (magnitude > 0)
		{
			this->x /= magnitude;
			this->y /= magnitude;
			this->z /= magnitude;
		}
	}

	bool Vector3::operator==(const Vector3 & other) const
	{
		if (this->x == other.x && this->y == other.y && this->z == other.z)
			return true;
		else
			return false;
	}

	bool Vector3::operator!=(const Vector3 & other) const
	{
		if (this->x == other.x && this->y == other.y && this->z == other.z)
			return false;
		else
			return true;
	}

	void Vector3::Clear()
	{
		this->x = 0;
		this->y = 0;
		this->z = 0;

		for (auto i = 0; i < 3; i++)
		{
			f[i] = 0;
		}
	}

	Vector3 Vector3::Slerp(float fact, const Vector3 & r) const
	{
		Vector3 v0 = *this;
		// Dot product - the cosine of the angle between 2 vectors.
		float dot = Dot(r);

		const double DOT_THRESHOLD = 0.9995;
		if (dot > DOT_THRESHOLD)
		{
			// If the inputs are too close for comfort, linearly interpolate
			// and normalize the result.

			Vector3 result = v0 + fact*(r - v0);
			result.Normalize();
			return result;
		}

		// Clamp it to be in the range of Acos()
		if (dot < -1.0f)
			dot = -1.0f;
		else if (dot > 1.0f)
			dot = 1.0f;

		// Acos(dot) returns the angle between start and end,
		// And multiplying that by percent returns the angle betweend
		// start and the final result.
		float theta_0 = acos(dot);
		float theta = acos(dot) * fact;

		Vector3 RelativeVec = r - v0*dot;
		RelativeVec.Normalize();     // Orthonormal basis

		return v0 * cos(theta) + RelativeVec * sin(theta);
	}

	Vector4::Vector4()
	{
		x, y, z, w = 0;
		//UpdateArray();
	}

	Vector4::Vector4(float x, float y, float z, float w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
		//UpdateArray();
	}

	void Vector4::UpdateArray()
	{
		f[0] = x;
		f[1] = y;
		f[2] = z;
		f[3] = w;
	}

	Vector4 Vector4::operator+(const Vector4 & v) const
	{
		Vector4 vector;
		vector.x = this->x + v.x;
		vector.y = this->y + v.y;
		vector.z = this->z + v.z;
		vector.w = this->w + v.w;

		return vector;
	}

	Vector4 Vector4::operator-(const Vector4 & v) const
	{
		Vector4 vector;
		vector.x = this->x - v.x;
		vector.y = this->y - v.y;
		vector.z = this->z - v.z;
		vector.w = this->w - v.w;

		return vector;
	}

	Vector4 Vector4::operator*(const float value) const
	{
		Vector4 vector;
		vector.x = this->x * value;
		vector.y = this->y * value;
		vector.z = this->z * value;
		vector.w = this->w * value;

		return vector;
	}

	Vector4 Vector4::operator/(const float value) const
	{
		Vector4 vector;
		vector.x = this->x / value;
		vector.y = this->y / value;
		vector.z = this->z / value;
		vector.w = this->w / value;

		return vector;
	}

	Vector4 Vector4::operator-() const
	{
		Vector4 vector;
		vector.x = this->x * -1;
		vector.y = this->y * -1;
		vector.z = this->z * -1;
		vector.w = this->w * -1;

		return vector;
	}

	float Vector4::operator*(const Vector4 & vector) const
	{
		return (this->x * vector.x) + (this->y * vector.y) + (this->z + vector.z) + (this->w + vector.w);
	}

	void Vector4::operator+=(const Vector4 & v)
	{
		this->x += v.x;
		this->y += v.y;
		this->z += v.z;
		this->w += v.w;
	}

	void Vector4::operator-=(const Vector4 & v)
	{
		this->x -= v.x;
		this->y -= v.y;
		this->z -= v.z;
		this->w -= v.w;
	}

	void Vector4::operator*=(const float value)
	{
		this->x *= value;
		this->y *= value;
		this->z *= value;
		this->w *= value;
	}
	/*
	Vector4 Vector4::Cross(const Vector4 & vector) const
	{
	Vector4 vec;

	vec.x = (this->y * vector.z) - (this->z * vector.y);
	vec.y = (this->z * vector.x) - (this->x * vector.z);
	vec.z = (this->x * vector.y) - (this->y * vector.x);

	return vec;
	}
	*/
	float Vector4::Dot(const Vector4 & vector) const
	{
		return (this->x * vector.x) + (this->y * vector.y) + (this->z * vector.z) + (this->w * vector.w);
	}

	float Vector4::Magnitude() const
	{
		return sqrt((x * x) + (y * y) + (z * z) + (w * w));
	}

	float Vector4::SquareMagnitude() const
	{
		return (pow(this->x, 2) + pow(this->y, 2) + pow(this->z, 2) + pow(this->w, 2));
	}

	void Vector4::Normalize()
	{
		float magnitude = Magnitude();
		if (magnitude > 0)
		{
			this->x /= magnitude;
			this->y /= magnitude;
			this->z /= magnitude;
			this->w /= magnitude;
		}
	}

	bool Vector4::operator==(const Vector4 & other) const
	{
		if (this->x == other.x && this->y == other.y && this->z == other.z && this->w == other.w)
			return true;
		else
			return false;
	}

	bool Vector4::operator!=(const Vector4 & other) const
	{
		if (this->x == other.x && this->y == other.y && this->z == other.z && this->w == other.w)
			return false;
		else
			return true;
	}

	void Vector4::Clear()
	{
		this->x = 0;
		this->y = 0;
		this->z = 0;
		this->w = 0;

		for (auto i = 0; i < 4; i++)
		{
			f[i] = 0;
		}
	}

	Vector4 Vector4::Slerp(float fact, const Vector4 & r) const
	{
		Vector4 v0 = *this;
		// Dot product - the cosine of the angle between 2 vectors.
		float dot = Dot(r);

		const double DOT_THRESHOLD = 0.9995;
		if (dot > DOT_THRESHOLD)
		{
			// If the inputs are too close for comfort, linearly interpolate
			// and normalize the result.

			Vector4 result = v0 + fact*(r - v0);
			result.Normalize();
			return result;
		}

		// Clamp it to be in the range of Acos()
		if (dot < -1.0f)
			dot = -1.0f;
		else if (dot > 1.0f)
			dot = 1.0f;

		// Acos(dot) returns the angle between start and end,
		// And multiplying that by percent returns the angle betweend
		// start and the final result.
		float theta_0 = acos(dot);
		float theta = acos(dot) * fact;

		Vector4 RelativeVec = r - v0*dot;
		RelativeVec.Normalize();     // Orthonormal basis

		return v0 * cos(theta) + RelativeVec * sin(theta);
	}

	Vector2 ToVector2(const Vector3 & vec)
	{
		return Vector2(vec.x, vec.y);
	}

	Vector3 ToVector3(const Vector2 & vec, float z)
	{
		return Vector3(vec.x, vec.y, z);
	}

	Vector3 ToVector3(const Vector2 & vec)
	{
		return Vector3(vec.x, vec.y, 0.0f);
	}

	Matrix44::Matrix44(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33)
	{

		float arr_m[4][4] = { m00, m01, m02, m03,
			m10, m11, m12, m13,
			m20, m21, m22, m23,
			m30, m31, m32, m33 };

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				m[i][j] = arr_m[i][j];
			}
		}
	}

	Matrix44::Matrix44()
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				m[i][j] = 0;
			}
		}

		m[0][0] = 1, m[1][1] = 1, m[2][2] = 1, m[3][3] = 1;
	}

	float Matrix44::Determinant33(float m11, float m12, float m13, float m21, float m22, float m23, float m31, float m32, float m33) const
	{
		return	(m11 * m22 * m33) +
			(m12 * m23 * m31) +
			(m13 * m21 * m32) -
			(m13 * m22 * m31) -
			(m11 * m23 * m32) -
			(m12 * m21 * m33);
	}

	/* Broken
	void Matrix44::UpdateAxisVectors()
	{
	xAxis = Vector3(m[0][0], m[1][0], m[2][0]);
	wx = m[3][0];
	yAxis = Vector3(m[0][1], m[1][1], m[2][1]);
	wy = m[3][1];
	zAxis = Vector3(m[0][2], m[1][2], m[2][2]);
	wz = m[3][2];

	xAxisH = Vector3(m[0][0], m[0][1], m[0][2]);
	wxh = m[0][3];
	yAxisH = Vector3(m[1][0], m[1][1], m[1][2]);
	wyh = m[1][3];
	zAxisH = Vector3(m[2][0], m[2][1], m[2][2]);
	wzh = m[2][3];
	}
	*/
	Vector3 Matrix44::operator*(const Vector3 & vec) const
	{
		Vector3 vector;
		vector.x = vec.Dot(Vector3(m[0][0], m[1][0], m[2][0])) + m[3][0];
		vector.y = vec.Dot(Vector3(m[0][1], m[1][1], m[2][1])) + m[3][1];
		vector.z = vec.Dot(Vector3(m[0][2], m[1][2], m[2][2])) + m[3][2];

		return vector;
	}

	Matrix44 Matrix44::operator+(const Matrix44 & mat) const
	{
		Matrix44 matrix;
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				matrix.m[i][j] = this->m[i][j] + mat.m[i][j];
			}
		}
		return matrix;
	}

	Matrix44 Matrix44::operator-(const Matrix44 & mat) const
	{
		Matrix44 matrix;
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				matrix.m[i][j] = this->m[i][j] - mat.m[i][j];
			}
		}

		return matrix;
	}

	Matrix44 Matrix44::operator*(const Matrix44 & mat) const
	{
		Matrix44 matrix(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

		for (int row = 0; row < 4; row++)
		{
			for (int column = 0; column < 4; column++)
			{
				for (int inner = 0; inner < 4; inner++)
				{
					matrix.m[row][column] += mat.m[row][inner] * m[inner][column];
				}
			}
		}
		return matrix;
	}

	Vector3 Matrix44::GetTranslation() const
	{
		return Vector3(m[3][0], m[3][1], m[3][2]);
	}

	void Matrix44::SetTranslation(const Vector3 & vec)
	{
		m[3][0] = vec.x, m[3][1] = vec.y, m[3][2] = vec.z;
	}

	Vector3 Matrix44::GetXAxis() const
	{
		return xAxis;
	}

	Vector3 Matrix44::GetYAxis() const
	{
		return yAxis;
	}

	Vector3 Matrix44::GetZAxis() const
	{
		return zAxis;
	}

	float Matrix44::Determinant() const
	{

		float det0, det1, det2, det3;

		det0 = m[0][0] * Determinant33(m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]);
		det1 = m[0][1] * Determinant33(m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]);
		det2 = m[0][2] * Determinant33(m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3]);
		det3 = m[0][3] * Determinant33(m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]);

		return det0 - det1 + det2 - det3;
	}

	bool Matrix44::Invert()
	{

		Matrix44 inverse;

		inverse.f[0] = f[5] * f[10] * f[15] -
			f[5] * f[11] * f[14] -
			f[9] * f[6] * f[15] +
			f[9] * f[7] * f[14] +
			f[13] * f[6] * f[11] -
			f[13] * f[7] * f[10];

		inverse.f[4] = -f[4] * f[10] * f[15] +
			f[4] * f[11] * f[14] +
			f[8] * f[6] * f[15] -
			f[8] * f[7] * f[14] -
			f[12] * f[6] * f[11] +
			f[12] * f[7] * f[10];

		inverse.f[8] = f[4] * f[9] * f[15] -
			f[4] * f[11] * f[13] -
			f[8] * f[5] * f[15] +
			f[8] * f[7] * f[13] +
			f[12] * f[5] * f[11] -
			f[12] * f[7] * f[9];

		inverse.f[12] = -f[4] * f[9] * f[14] +
			f[4] * f[10] * f[13] +
			f[8] * f[5] * f[14] -
			f[8] * f[6] * f[13] -
			f[12] * f[5] * f[10] +
			f[12] * f[6] * f[9];

		inverse.f[1] = -f[1] * f[10] * f[15] +
			f[1] * f[11] * f[14] +
			f[9] * f[2] * f[15] -
			f[9] * f[3] * f[14] -
			f[13] * f[2] * f[11] +
			f[13] * f[3] * f[10];

		inverse.f[5] = f[0] * f[10] * f[15] -
			f[0] * f[11] * f[14] -
			f[8] * f[2] * f[15] +
			f[8] * f[3] * f[14] +
			f[12] * f[2] * f[11] -
			f[12] * f[3] * f[10];

		inverse.f[9] = -f[0] * f[9] * f[15] +
			f[0] * f[11] * f[13] +
			f[8] * f[1] * f[15] -
			f[8] * f[3] * f[13] -
			f[12] * f[1] * f[11] +
			f[12] * f[3] * f[9];

		inverse.f[13] = f[0] * f[9] * f[14] -
			f[0] * f[10] * f[13] -
			f[8] * f[1] * f[14] +
			f[8] * f[2] * f[13] +
			f[12] * f[1] * f[10] -
			f[12] * f[2] * f[9];

		inverse.f[2] = f[1] * f[6] * f[15] -
			f[1] * f[7] * f[14] -
			f[5] * f[2] * f[15] +
			f[5] * f[3] * f[14] +
			f[13] * f[2] * f[7] -
			f[13] * f[3] * f[6];

		inverse.f[6] = -f[0] * f[6] * f[15] +
			f[0] * f[7] * f[14] +
			f[4] * f[2] * f[15] -
			f[4] * f[3] * f[14] -
			f[12] * f[2] * f[7] +
			f[12] * f[3] * f[6];

		inverse.f[10] = f[0] * f[5] * f[15] -
			f[0] * f[7] * f[13] -
			f[4] * f[1] * f[15] +
			f[4] * f[3] * f[13] +
			f[12] * f[1] * f[7] -
			f[12] * f[3] * f[5];

		inverse.f[14] = -f[0] * f[5] * f[14] +
			f[0] * f[6] * f[13] +
			f[4] * f[1] * f[14] -
			f[4] * f[2] * f[13] -
			f[12] * f[1] * f[6] +
			f[12] * f[2] * f[5];

		inverse.f[3] = -f[1] * f[6] * f[11] +
			f[1] * f[7] * f[10] +
			f[5] * f[2] * f[11] -
			f[5] * f[3] * f[10] -
			f[9] * f[2] * f[7] +
			f[9] * f[3] * f[6];

		inverse.f[7] = f[0] * f[6] * f[11] -
			f[0] * f[7] * f[10] -
			f[4] * f[2] * f[11] +
			f[4] * f[3] * f[10] +
			f[8] * f[2] * f[7] -
			f[8] * f[3] * f[6];

		inverse.f[11] = -f[0] * f[5] * f[11] +
			f[0] * f[7] * f[9] +
			f[4] * f[1] * f[11] -
			f[4] * f[3] * f[9] -
			f[8] * f[1] * f[7] +
			f[8] * f[3] * f[5];

		inverse.f[15] = f[0] * f[5] * f[10] -
			f[0] * f[6] * f[9] -
			f[4] * f[1] * f[10] +
			f[4] * f[2] * f[9] +
			f[8] * f[1] * f[6] -
			f[8] * f[2] * f[5];

		float det0 = f[0] * inverse.f[0] + f[1] * inverse.f[4] + f[2] * inverse.f[8] + f[3] * inverse.f[12];

		if (det0 == 0)
			return false;

		det0 = 1.0f / det0;

		for (int i = 0; i < 16; i++)
			f[i] = inverse.f[i] * det0;

		return true;
	}

	void Matrix44::Transpose()
	{
		Matrix44 mat;
		for (auto i = 0; i < 4; ++i)
		{
			for (auto j = 0; j < 4; ++j)
			{
				mat.m[j][i] = m[i][j];
			}
		}
		*this = mat;
	}

	void Matrix44::SetOrientation(const Vector3 & x, const Vector3 & y, const Vector3 & z)
	{
		m[0][0] = x.x, m[0][1] = x.y, m[0][2] = x.z;
		m[1][0] = y.x, m[1][1] = y.y, m[1][2] = y.z;
		m[2][0] = z.x, m[2][1] = z.y, m[2][2] = z.z;
	}

	void Matrix44::SetEulerAxis(float yaw, float pitch, float roll)
	{
		*this = CreateIdentity();
		m[0][0] = cosf(yaw) * cosf(pitch) - cosf(roll) * sinf(pitch) * sinf(yaw);
		m[0][1] = cosf(yaw) * sinf(pitch) + cosf(roll) * cosf(pitch) * sinf(yaw);
		m[0][2] = sinf(yaw) * sinf(roll);

		m[1][0] = -sinf(yaw) * cosf(pitch) - cosf(roll) * sinf(pitch) * cosf(yaw);
		m[1][1] = -sinf(yaw) * sinf(pitch) + cosf(roll) * cosf(pitch) * cosf(yaw);
		m[1][2] = cosf(yaw) * sinf(roll);

		m[2][0] = sinf(roll) * sinf(pitch);
		m[2][1] = -sinf(roll) * cosf(pitch);
		m[2][2] = cosf(roll);
		Transpose();
	}

	Matrix44 Matrix44::CreateIdentity()
	{

		Matrix44 matrix;
		return matrix;
	}

	Matrix44 Matrix44::CreateTranslation(float x, float y, float z)
	{
		Matrix44 matrix;
		matrix.m[3][0] = x, matrix.m[3][1] = y, matrix.m[3][2] = z;

		return matrix;
	}

	Matrix44 Matrix44::CreateScale(Vector3 scale)
	{
		Matrix44 matrix;
		matrix.m[0][0] = scale.x, matrix.m[1][1] = scale.y, matrix.m[2][2] = scale.z;

		return matrix;
	}

	Matrix44 Matrix44::CreateRotate(float angle, const Vector3 & axis)
	{
		float pCos = cos(angle);
		float pSin = sin(angle);
		float _pCosN = 1.0f - pCos;

		Matrix44 matrix(pCos + _pCosN * axis.f[0] * axis.f[0],
			_pCosN * axis.f[0] * axis.f[1] + axis.f[2] * pSin,
			_pCosN * axis.f[0] * axis.f[2] - axis.f[1] * pSin,
			0.0f, _pCosN * axis.f[0] * axis.f[1] - axis.f[2] * pSin,
			pCos + _pCosN * axis.f[1] * axis.f[1],
			_pCosN * axis.f[1] * axis.f[2] + axis.f[0] * pSin,
			0.0f, _pCosN * axis.f[0] * axis.f[2] + axis.f[1] * pSin,
			_pCosN * axis.f[1] * axis.f[2] - axis.f[0] * pSin,
			pCos + _pCosN * axis.f[2] * axis.f[2],
			0.0f, 0.0f, 0.0f, 0.0f, 1.0f);


		return matrix;

	}

	Matrix44 Matrix44::CreateRotateX(float angle)
	{
		Matrix44 matrix(1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, cos(angle), sin(angle), 0.0f,
			0.0f, -sin(angle), cos(angle), 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);

		return matrix;
	}

	Matrix44 Matrix44::CreateRotateY(float angle)
	{
		float pCos = cos(angle);
		float pSin = sin(angle);
		Matrix44 matrix(cos(angle), 0.0f, -sin(angle), 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			sin(angle), 0.0f, cos(angle), 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);

		return matrix;
	}

	Matrix44 Matrix44::CreateRotateZ(float angle)
	{
		Matrix44 matrix(cos(angle), sin(angle), 0.0f, 0.0f,
			-sin(angle), cos(angle), 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);

		return matrix;
	}

	Matrix44 Matrix44::CreateOrtho(float left, float right, float bottom, float top, float nearZ, float farZ)
	{
		Matrix44    projection;
		float       deltaX = right - left;
		float       deltaY = top - bottom;
		float       deltaZ = farZ - nearZ;

		if ((deltaX == 0.0f) || (deltaY == 0.0f) || (deltaZ == 0.0f))
			return projection;

		projection.m[0][0] = 2.0f / deltaX;
		projection.m[1][1] = 2.0f / deltaY;
		projection.m[2][2] = -2.0f / deltaZ;
		projection.m[3][0] = -(right + left) / deltaX;
		projection.m[3][1] = -(top + bottom) / deltaY;
		projection.m[3][2] = -(nearZ + farZ) / deltaZ;

		return projection;
	}

	Matrix44 Matrix44::CreateFrustum(float left, float right, float bottom, float top, float nearZ, float farZ)
	{
		Matrix44    frustum(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		float       deltaX = right - left;
		float       deltaY = top - bottom;
		float       deltaZ = farZ - nearZ;


		if ((nearZ <= 0.0f) || (farZ <= 0.0f) ||
			(deltaX <= 0.0f) || (deltaY <= 0.0f) || (deltaZ <= 0.0f))
			return CreateIdentity();;

		frustum.m[0][0] = 2.0f * nearZ / deltaX;

		frustum.m[1][1] = 2.0f * nearZ / deltaY;

		frustum.m[2][0] = (right + left) / deltaX;
		frustum.m[2][1] = (top + bottom) / deltaY;
		frustum.m[2][2] = -(nearZ + farZ) / deltaZ;
		frustum.m[2][3] = -1.0f;

		frustum.m[3][2] = -2.0f * nearZ * farZ / deltaZ;

		return frustum;
	}
	//frustum.m[0][1] = frustum.m[0][2] = frustum.m[0][3] = 0.0f;
	//frustum.m[1][0] = frustum.m[1][2] = frustum.m[1][3] = 0.0f;
	//frustum.m[3][0] = frustum.m[3][1] = frustum.m[3][3] = 0.0f;

	Matrix44 Matrix44::CreatePerspective(float fovy, float aspect, float nearZ, float farZ)
	{
		float cotan = 1.0f / tanf(fovy / 2.0f);

		Matrix44 matrix(cotan / aspect, 0.0f, 0.0f, 0.0f,
			0.0f, cotan, 0.0f, 0.0f,
			0.0f, 0.0f, (farZ + nearZ) / (nearZ - farZ), -1.0f,
			0.0f, 0.0f, (2.0f * farZ * nearZ) / (nearZ - farZ), 0.0f);
		return matrix;
	}

	Matrix44 Matrix44::CreateLookAt(const Vector3 & eye, const Vector3 & center, const Vector3 & up)
	{
		//Constructing the camera space 
		Vector3 w = eye - center;
		w.Normalize();

		Vector3 u = up.Cross(w);
		u.Normalize();

		Vector3 v = w.Cross(u);
		v.Normalize();

		Matrix44 rotation(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);

		Matrix44 translation = CreateTranslation(-eye.x, -eye.y, -eye.z);


		Matrix44 view = rotation * translation;

		return view;
	}

	Vector3 Matrix44::TransformDirectionVector(const Vector3 & direction)
	{
		Vector3 result(direction.x * m[0][0] +
			direction.y * m[1][0] +
			direction.z * m[2][0],

			direction.x * m[0][1] +
			direction.y * m[1][1] +
			direction.z * m[2][1],

			direction.x * m[0][2] +
			direction.y * m[1][2] +
			direction.z * m[2][2]);

		return result;
	}

}
