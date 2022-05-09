#pragma once

#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <boost/json/src.hpp>

namespace json = boost::json;

class GLTFLoader
{
public:
	GLTFLoader(const char* filename);
	~GLTFLoader();

	//std::string GetProperty(const char* property);

	const json::object* GetGLTF() const
	{
		return &m_gltf;
	}

	std::string GetBinaryName() const
	{
		return m_gltf.at("buffers").as_array()[0].at("uri").as_string().c_str();
	}

private:
	std::string m_gltfFilename;
	std::string m_binaryFilename;
	json::object m_gltf;
	std::string m_binary;
};

GLTFLoader::GLTFLoader(const char* filename)
	: m_gltfFilename(filename)
{
	std::filesystem::path path(filename);
	std::string gltfFile;
	if (std::filesystem::exists(path))
	{
		std::ifstream file(path);
		gltfFile = std::string(std::istreambuf_iterator<char>(file), {});
		file.close();
		m_gltf = json::parse(gltfFile).as_object();

		path = path.parent_path().append(GetBinaryName());

		file.open(path, std::ios_base::binary);
		m_binary = std::string(std::istreambuf_iterator<char>(file), {});
		file.close();
	}
	else
	{
		std::cerr << "gltf file is not exits";
	}

}

GLTFLoader::~GLTFLoader()
{
}

//std::string GLTFLoader::GetProperty(const char* property)
//{
//	if (m_gltf.empty())
//		return std::to_string(-1);
//	
//	auto obj = json::parse(m_gltf).as_object();
//	auto pValue = obj.if_contains(property);
//	int64_t value = -1;
//	if (pValue)
//	{
//		//value = pValue->as_int64();
//		value = obj.at(property).as_int64();
//	}
//	return std::to_string(value);
//}

