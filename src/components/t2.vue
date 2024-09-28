<template>
  <el-upload
      class="upload-demo"
      ref="upload2"
      action="http://localhost:5000"
      :on-preview="handlePreview"
      :on-remove="handleRemove"
      :file-list="fileList"
      :auto-upload="false"
      :limit="2"
      list-type="picture"
      :on-success="success">
    <el-button size="small" type="primary">点击上传</el-button>
    <div slot="tip" class="el-upload__tip">请在此处上传变化后的图像</div>
  </el-upload>
</template>

<script>
export default {
  data() {
    return {
      fileList: [],
    };
  },
  methods: {
    success(response, file, fileList){
      this.$emit('transmit', response)
      console.log(response)
    },
    handleRemove(file, fileList) {
      console.log(file, fileList);
    },
    handlePreview(file) {
      console.log(file);
    },
    submit(){
      this.$refs.upload2.submit();
    }
  }
}
</script>