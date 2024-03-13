#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#define WORLDUP glm::vec3(0.0f, 1.0f, 0.0f)


class Camera
{
public:
	Camera();

	Camera(const glm::vec3& origin, float vfov, float aspect, float aperture, float focalLength);

	static const glm::vec3& m_defaultForward;

	void UVWFrame(glm::vec3& U, glm::vec3& V, glm::vec3& W);

	inline float GetLensRadius() const { return m_aperture * 0.5f; }

	inline const glm::vec3& GetPosition() const { return m_position; }

	inline void SetForward(glm::vec3 direction) { m_forward = normalize(direction); }

	inline float GetFocalLength() { return m_focalLength; }
	inline void SetFocalLength(float length) 
	{ 
		if (m_focalLength != length)
		{
			m_focalLength = length; 
			m_changed = true;
		}
	}

	inline void MoveForward(float speed)
	{
		m_position += m_forward * speed;

		m_changed = true;
	}

	inline void MoveRight(float speed) {
		m_position += normalize(m_u) * speed; 

		m_changed = true;
	}

	inline void MoveUp(float speed)
	{
		m_position += normalize(m_v) * speed;
		
		m_changed = true;
	}

	void Pitch(float speed)
	{
		glm::quat rotation = glm::rotate(glm::quat(1, 0, 0, 0), glm::radians(speed), m_u);
		m_forward = normalize(m_forward * rotation);

		m_changed = true;
	}

	void Yaw(float speed)
	{
		glm::quat rotation = glm::rotate(glm::quat(1, 0, 0, 0), glm::radians(speed), m_v);
		m_forward = normalize(m_forward * rotation);

		m_changed = true;
	}

	void Roll(float speed)
	{
		glm::quat rotation = glm::rotate(glm::quat(1, 0, 0, 0), glm::radians(speed), m_w);
		m_forward = normalize(m_forward * rotation);

		m_changed = true;
	}

	bool Changed()
	{
		bool changed = m_changed;
		m_changed = false;
		return changed;
	}

private:
	glm::vec3 m_position = glm::vec3(0.0f);
	glm::vec3 m_forward  = m_defaultForward;
	float m_vfov         = 45.0f;
	float m_aspect       = 1.6f;
	float m_aperture     = 0.0f;
	float m_focalLength  = 1.0f;

	glm::vec3 m_u, m_v, m_w;

	void UpdateUVW();

	bool m_changed = true;
};
